"""Main window: file list on the left, image canvas in the middle, box
table on the right, plus the menus/toolbar that drive auto-labeling,
saving, undo/redo, and dataset export."""
from __future__ import annotations

import copy
import os
from typing import List, Optional, Set

import cv2

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                                QDockWidget, QFileDialog, QHBoxLayout,
                                QHeaderView, QLabel, QListWidget, QMainWindow,
                                QMessageBox, QProgressDialog, QPushButton,
                                QStatusBar, QTableWidget, QTableWidgetItem,
                                QToolBar, QVBoxLayout, QWidget)

from . import export as export_utils
from .canvas_view import CanvasView
from .label_store import LabelStore
from .ocr_engine import OCREngine
from .shapes import Shape
from .workers import OcrBatchWorker
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LANGUAGES = [
    ("en", "English"),
    ("ch", "Chinese (Simplified) + English"),
    ("chinese_cht", "Chinese (Traditional)"),
    ("japan", "Japanese"),
    ("korean", "Korean"),
    ("vi", "Vietnamese"),
    ("french", "French"),
    ("german", "German"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("arabic", "Arabic"),
    ("hi", "Hindi"),
    ("th", "Thai"),
    ("latin", "Latin (general)"),
    ("devanagari", "Devanagari (general)"),
    ("cyrillic", "Cyrillic (general)"),
]

UNDO_LIMIT = 50

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PaddleOCR Label Studio")
        self.resize(1440, 900)

        self.image_dir: Optional[str] = None
        self.image_list: List[str] = []
        self.current_index: int = -1
        self.current_shapes: List[Shape] = []
        self.label_store: Optional[LabelStore] = None
        self.touched: Set[str] = set()
        self.dirty: bool = False

        self.undo_stack: List[List[Shape]] = []
        self.redo_stack: List[List[Shape]] = []

        self.engine = OCREngine(lang="en")
        self.ocr_worker: Optional[OcrBatchWorker] = None
        self.progress_dialog: Optional[QProgressDialog] = None
        self._shown_ocr_error_dialog = False

        # Cache of the current image's pixels (BGR, for cv2-based cropping),
        # so re-recognizing several boxes in a row doesn't re-read the file
        # from disk each time. Invalidated in _enter_image().
        self._current_cv_image = None
        # Only nag once per session if auto re-recognize is on but paddleocr
        # isn't installed, instead of on every single box edit.
        self._auto_recognize_warned = False

        self._updating_selection = False  # re-entrancy guard for table<->canvas sync

        self._build_ui()
        self._build_actions_and_menus()
        self._connect_signals()
        self._update_title()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self) -> None:
        self.canvas = CanvasView()
        self.setCentralWidget(self.canvas)

        # ---- left dock: image list ----
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        left_dock = QDockWidget("Images", self)
        left_dock.setObjectName("images_dock")
        left_dock.setWidget(self.file_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        # ---- right dock: language picker + box table ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("OCR language:"))
        self.lang_combo = QComboBox()
        for code, name in LANGUAGES:
            self.lang_combo.addItem(name, code)
        right_layout.addWidget(self.lang_combo)

        draw_row = QHBoxLayout()
        self.btn_draw_rect = QPushButton("Draw box")
        self.btn_draw_rect.setToolTip("Click-drag on the image to add an axis-aligned box (Ctrl+B)")
        self.btn_draw_quad = QPushButton("Draw quad")
        self.btn_draw_quad.setToolTip("Click 4 corners in order to add a rotated/skewed box (Ctrl+Shift+B)")
        draw_row.addWidget(self.btn_draw_rect)
        draw_row.addWidget(self.btn_draw_quad)
        right_layout.addLayout(draw_row)

        right_layout.addWidget(QLabel("Text boxes in this image:"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Text", "Difficult", ""])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 70)
        self.table.setColumnWidth(2, 30)
        right_layout.addWidget(self.table)

        right_dock = QDockWidget("Text boxes", self)
        right_dock.setObjectName("boxes_dock")
        right_dock.setWidget(right_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        self.setStatusBar(QStatusBar())

    def _build_actions_and_menus(self) -> None:
        menubar = self.menuBar()

        # ---------------- File ----------------
        file_menu = menubar.addMenu("&File")

        act_open = QAction("&Open Image Folder...", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_folder)
        file_menu.addAction(act_open)

        act_save = QAction("&Save", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self.save_current_image_labels)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_export_rec = QAction("Export Recognition Dataset...", self)
        act_export_rec.triggered.connect(self.export_recognition_dataset)
        file_menu.addAction(act_export_rec)

        act_export_det = QAction("Export Detection Train/Val Split...", self)
        act_export_det.triggered.connect(self.export_detection_split)
        file_menu.addAction(act_export_det)

        act_build_dict = QAction("Build Character Dictionary...", self)
        act_build_dict.triggered.connect(self.build_char_dict)
        file_menu.addAction(act_build_dict)

        file_menu.addSeparator()

        act_quit = QAction("E&xit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # ---------------- Edit ----------------
        edit_menu = menubar.addMenu("&Edit")

        self.act_undo = QAction("&Undo", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(self.undo)
        edit_menu.addAction(self.act_undo)

        self.act_redo = QAction("&Redo", self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.triggered.connect(self.redo)
        edit_menu.addAction(self.act_redo)

        edit_menu.addSeparator()

        act_draw_rect = QAction("Draw Box Mode", self)
        act_draw_rect.setShortcut(QKeySequence("Ctrl+B"))
        act_draw_rect.triggered.connect(lambda: self.canvas.set_mode(CanvasView.MODE_RECT))
        edit_menu.addAction(act_draw_rect)

        act_draw_quad = QAction("Draw Quad Mode", self)
        act_draw_quad.setShortcut(QKeySequence("Ctrl+Shift+B"))
        act_draw_quad.triggered.connect(lambda: self.canvas.set_mode(CanvasView.MODE_QUAD))
        edit_menu.addAction(act_draw_quad)

        # ---------------- Navigate ----------------
        nav_menu = menubar.addMenu("&Navigate")

        act_next = QAction("&Next Image", self)
        act_next.setShortcut(QKeySequence("PgDown"))
        act_next.triggered.connect(self.next_image)
        nav_menu.addAction(act_next)

        act_prev = QAction("&Previous Image", self)
        act_prev.setShortcut(QKeySequence("PgUp"))
        act_prev.triggered.connect(self.prev_image)
        nav_menu.addAction(act_prev)

        # ---------------- OCR ----------------
        ocr_menu = menubar.addMenu("&OCR")

        act_label_current = QAction("Auto-Label &Current Image", self)
        act_label_current.triggered.connect(self.auto_label_current)
        ocr_menu.addAction(act_label_current)

        act_label_all = QAction("Auto-Label &All Images...", self)
        act_label_all.triggered.connect(self.auto_label_all)
        ocr_menu.addAction(act_label_all)

        act_rerecognize = QAction("Re-recognize Selected Box", self)
        act_rerecognize.triggered.connect(self.recognize_selected_box)
        ocr_menu.addAction(act_rerecognize)

        self.act_auto_recognize = QAction("Auto Re-recognize After Editing Box", self)
        self.act_auto_recognize.setCheckable(True)
        self.act_auto_recognize.setChecked(True)
        self.act_auto_recognize.setToolTip(
            "When checked, dragging a corner or moving a box automatically "
            "re-runs recognition on its new crop and updates the text.\n"
            "Turn this off if you'd rather hand-correct text without a "
            "fresh OCR guess overwriting it."
        )
        ocr_menu.addAction(self.act_auto_recognize)

        # ---------------- View ----------------
        view_menu = menubar.addMenu("&View")

        act_zoom_in = QAction("Zoom In", self)
        act_zoom_in.setShortcut(QKeySequence.ZoomIn)
        act_zoom_in.triggered.connect(lambda: self.canvas.zoom_by(1.15))
        view_menu.addAction(act_zoom_in)

        act_zoom_out = QAction("Zoom Out", self)
        act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        act_zoom_out.triggered.connect(lambda: self.canvas.zoom_by(1 / 1.15))
        view_menu.addAction(act_zoom_out)

        act_fit = QAction("Fit to Window", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(act_fit)

        # ---------------- Help ----------------
        help_menu = menubar.addMenu("&Help")
        act_shortcuts = QAction("Keyboard Shortcuts", self)
        act_shortcuts.triggered.connect(self.show_shortcuts)
        help_menu.addAction(act_shortcuts)

        # ---------------- toolbar (quick access) ----------------
        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("main_toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(act_open)
        toolbar.addAction(act_save)
        toolbar.addSeparator()
        toolbar.addAction(act_label_current)
        toolbar.addAction(act_label_all)
        toolbar.addSeparator()
        toolbar.addAction(act_prev)
        toolbar.addAction(act_next)
        toolbar.addSeparator()
        toolbar.addAction(self.act_undo)
        toolbar.addAction(self.act_redo)

    def _connect_signals(self) -> None:
        self.file_list.currentRowChanged.connect(self.on_file_list_row_changed)

        self.canvas.shapesChanged.connect(self.on_canvas_shapes_changed)
        self.canvas.shapeCreated.connect(self.on_canvas_shape_created)
        self.canvas.selectionChanged.connect(self.on_canvas_selection_changed)
        self.canvas.dragStarted.connect(self.push_undo)
        self.canvas.deleteRequested.connect(self.delete_shape_at)

        self.table.itemChanged.connect(self.on_table_item_changed)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)

        self.btn_draw_rect.clicked.connect(lambda: self.canvas.set_mode(CanvasView.MODE_RECT))
        self.btn_draw_quad.clicked.connect(lambda: self.canvas.set_mode(CanvasView.MODE_QUAD))

        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)

    # ==================================================================
    # folder loading / navigation
    # ==================================================================
    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self.load_folder(path)

    def load_folder(self, path: str) -> None:
        # Flush any unsaved edits to the folder we're about to leave before
        # swapping in a new LabelStore, or they'd be silently discarded.
        if self.dirty:
            self.save_current_image_labels()

        self.image_dir = path
        self.label_store = LabelStore(path)
        self.touched = set(self.label_store.data.keys())

        names = sorted(
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        self.image_list = names
        self.current_index = -1
        self.current_shapes = []

        self.file_list.blockSignals(True)
        self.file_list.clear()
        for n in names:
            self.file_list.addItem(self._make_list_label(n))
        self.file_list.blockSignals(False)

        if names:
            self.file_list.setCurrentRow(0)
        else:
            QMessageBox.information(self, "No images found",
                                     "That folder doesn't contain any supported image files.")
        self._update_title()

    def _make_list_label(self, relpath: str) -> str:
        n = len(self.label_store.get(relpath)) if self.label_store else 0
        prefix = "\u2713 " if relpath in self.touched else "   "
        suffix = f"  ({n})" if n else ""
        return f"{prefix}{relpath}{suffix}"

    def _refresh_list_item(self, relpath: str) -> None:
        if relpath in self.image_list:
            idx = self.image_list.index(relpath)
            self.file_list.item(idx).setText(self._make_list_label(relpath))

    @Slot(int)
    def on_file_list_row_changed(self, row: int) -> None:
        if row < 0 or row == self.current_index:
            return
        self._leave_current_image()
        self.current_index = row
        self._enter_image(row)

    def _leave_current_image(self) -> None:
        if self.current_index < 0:
            return
        if self.dirty:
            self.save_current_image_labels()

    def _enter_image(self, index: int) -> None:
        relpath = self.image_list[index]
        full_path = os.path.join(self.image_dir, relpath)
        pixmap = QPixmap(full_path)
        self.canvas.load_image(pixmap)
        self._current_cv_image = None  # new image on screen -> drop the stale cv2 cache

        stored = self.label_store.get(relpath) if self.label_store else []
        self.current_shapes = [copy.deepcopy(s) for s in stored]
        self.canvas.set_shapes(self.current_shapes)
        self.canvas.fit_to_window()
        self._rebuild_table()

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self._shown_ocr_error_dialog = False
        self.statusBar().showMessage(f"{relpath}   ({index + 1}/{len(self.image_list)})")

    def next_image(self) -> None:
        if self.current_index + 1 < len(self.image_list):
            self.file_list.setCurrentRow(self.current_index + 1)

    def prev_image(self) -> None:
        if self.current_index - 1 >= 0:
            self.file_list.setCurrentRow(self.current_index - 1)

    # ==================================================================
    # table <-> shape-list <-> canvas synchronization
    # ==================================================================
    def _rebuild_table(self) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for shape in self.current_shapes:
            self._append_table_row(shape)
        self.table.blockSignals(False)

    def _append_table_row(self, shape: Shape) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        text_item = QTableWidgetItem(shape.transcription)
        self.table.setItem(row, 0, text_item)

        chk = QCheckBox()
        chk.setChecked(shape.difficult)
        chk.stateChanged.connect(lambda state, c=chk: self.on_difficult_toggled(c, state))
        self.table.setCellWidget(row, 1, chk)

        del_btn = QPushButton("\u2715")
        del_btn.setFixedWidth(26)
        del_btn.clicked.connect(lambda _, b=del_btn: self.on_delete_button_clicked(b))
        self.table.setCellWidget(row, 2, del_btn)

    def _row_of_widget(self, widget) -> int:
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 1) is widget or self.table.cellWidget(row, 2) is widget:
                return row
        return -1

    @Slot(QTableWidgetItem)
    def on_table_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if item.column() == 0 and 0 <= row < len(self.current_shapes):
            self.current_shapes[row].transcription = item.text()
            self.mark_dirty()

    def on_difficult_toggled(self, checkbox: QCheckBox, state: int) -> None:
        row = self._row_of_widget(checkbox)
        if 0 <= row < len(self.current_shapes):
            value = bool(state)
            self.current_shapes[row].difficult = value
            self.canvas.set_shape_difficult(row, value)
            self.mark_dirty()

    def on_delete_button_clicked(self, button: QPushButton) -> None:
        row = self._row_of_widget(button)
        if row >= 0:
            self.delete_shape_at(row)

    def on_table_selection_changed(self) -> None:
        if self._updating_selection:
            return
        rows = self.table.selectionModel().selectedRows()
        self._updating_selection = True
        self.canvas.select_shape(rows[0].row() if rows else -1)
        self._updating_selection = False

    @Slot(int)
    def on_canvas_selection_changed(self, index: int) -> None:
        if self._updating_selection:
            return
        self._updating_selection = True
        if index < 0:
            self.table.clearSelection()
        else:
            self.table.selectRow(index)
        self._updating_selection = False

    # ==================================================================
    # canvas edits -> data model
    # ==================================================================
    @Slot(int)
    def on_canvas_shapes_changed(self, changed_index: int) -> None:
        for i, pts in enumerate(self.canvas.get_shapes_points()):
            if i < len(self.current_shapes):
                self.current_shapes[i].points = pts
        self.mark_dirty()
        if self.act_auto_recognize.isChecked():
            # The undo snapshot for this edit was already pushed in
            # push_undo() when the drag started (see dragStarted), so this
            # follow-up transcription update rides along in the same undo
            # step rather than adding a second one.
            self._recognize_box(changed_index, show_errors=False, push_undo=False)

    @Slot(list)
    def on_canvas_shape_created(self, points) -> None:
        self.push_undo()
        shape = Shape(points=[tuple(p) for p in points], transcription="", difficult=False)
        self.current_shapes.append(shape)
        self.canvas.add_shape(points, difficult=False, select=True)
        self._rebuild_table()
        self.table.selectRow(len(self.current_shapes) - 1)
        self.mark_dirty()

    @Slot(int)
    def delete_shape_at(self, index: int) -> None:
        if not (0 <= index < len(self.current_shapes)):
            return
        self.push_undo()
        del self.current_shapes[index]
        self.canvas.remove_shape(index)
        self._rebuild_table()
        self.mark_dirty()

    # ==================================================================
    # undo / redo -- snapshots of the current image's shape list
    # ==================================================================
    def push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.current_shapes))
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.current_shapes))
        self.current_shapes = self.undo_stack.pop()
        self._apply_shapes_to_views()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.current_shapes))
        self.current_shapes = self.redo_stack.pop()
        self._apply_shapes_to_views()

    def _apply_shapes_to_views(self) -> None:
        self.canvas.set_shapes(self.current_shapes)
        self._rebuild_table()
        self.mark_dirty()

    # ==================================================================
    # persistence
    # ==================================================================
    def mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def save_current_image_labels(self) -> None:
        if self.current_index < 0 or self.label_store is None:
            return
        relpath = self.image_list[self.current_index]
        self.label_store.set(relpath, self.current_shapes)
        self.label_store.save()
        self.touched.add(relpath)
        self.dirty = False
        self._refresh_list_item(relpath)
        self._update_title()
        self.statusBar().showMessage(f"Saved {relpath}", 3000)

    def _update_title(self) -> None:
        base = "PaddleOCR Label Studio"
        if self.image_dir:
            base += f" \u2014 {self.image_dir}"
        if self.dirty:
            base += " *"
        self.setWindowTitle(base)

    def closeEvent(self, event) -> None:
        if self.dirty:
            self.save_current_image_labels()
        super().closeEvent(event)

    # ==================================================================
    # OCR
    # ==================================================================
    def on_language_changed(self, _index: int) -> None:
        code = self.lang_combo.currentData()
        self.engine = OCREngine(lang=code)

    def auto_label_current(self) -> None:
        if self.current_index < 0:
            QMessageBox.information(self, "No image", "Open an image folder first.")
            return
        self._run_ocr_batch([self.image_list[self.current_index]])

    def auto_label_all(self) -> None:
        if not self.image_list:
            QMessageBox.information(self, "No images", "Open an image folder first.")
            return
        choice = QMessageBox.question(
            self, "Auto-label all images",
            f"Run OCR on all {len(self.image_list)} images?\n\n"
            "Yes = skip images that already have saved boxes.\n"
            "No = re-label every image, overwriting existing boxes.",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Cancel:
            return
        targets = self.image_list if choice == QMessageBox.No else \
            [r for r in self.image_list if r not in self.touched]
        if not targets:
            QMessageBox.information(self, "Nothing to do", "All images are already labeled.")
            return
        self._run_ocr_batch(targets)

    def _run_ocr_batch(self, relpaths: List[str]) -> None:
        if self.dirty:
            self.save_current_image_labels()

        self._shown_ocr_error_dialog = False
        self.progress_dialog = QProgressDialog("Starting OCR...", "Cancel", 0, len(relpaths), self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        self.ocr_worker = OcrBatchWorker(self.engine, self.image_dir, relpaths)
        self.ocr_worker.progress.connect(self._on_ocr_progress)
        self.ocr_worker.image_result.connect(self._on_ocr_image_result)
        self.ocr_worker.error.connect(self._on_ocr_error)
        self.ocr_worker.finished_all.connect(self._on_ocr_finished)
        self.progress_dialog.canceled.connect(self.ocr_worker.cancel)
        self.ocr_worker.start()

    def _on_ocr_progress(self, current: int, total: int, filename: str) -> None:
        if self.progress_dialog:
            self.progress_dialog.setLabelText(f"OCR: {filename} ({current}/{total})")
            self.progress_dialog.setValue(current)

    def _on_ocr_image_result(self, relpath: str, shape_dicts: list) -> None:
        shapes = [
            Shape(points=[tuple(p) for p in d["points"]],
                  transcription=d.get("transcription", ""),
                  difficult=False, score=d.get("score", 1.0))
            for d in shape_dicts
        ]
        self.label_store.set(relpath, shapes)
        self.touched.add(relpath)
        self._refresh_list_item(relpath)
        if 0 <= self.current_index < len(self.image_list) and self.image_list[self.current_index] == relpath:
            self.current_shapes = shapes
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.canvas.set_shapes(self.current_shapes)
            self._rebuild_table()
            # Freshly-applied OCR results are already written into label_store
            # above, so there's nothing unsaved yet -- don't mark dirty.
            self.dirty = False
            self._update_title()

    def _on_ocr_error(self, relpath: str, message: str) -> None:
        self.statusBar().showMessage(f"OCR error on {relpath}: {message}", 8000)
        if not self._shown_ocr_error_dialog:
            self._shown_ocr_error_dialog = True
            QMessageBox.warning(self, "OCR error", f"{relpath}:\n{message}")

    def _on_ocr_finished(self) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        if self.label_store:
            self.label_store.save()
        self.statusBar().showMessage("Auto-labeling finished.", 5000)

    def recognize_selected_box(self) -> None:
        index = self.canvas.selected_index
        if index < 0 or not (0 <= index < len(self.current_shapes)):
            QMessageBox.information(self, "No box selected", "Select a box first.")
            return
        self._recognize_box(index, show_errors=True, push_undo=True)

    def _get_current_cv_image(self):
        """Return the current image as a BGR array for crop-based
        recognition, loading it from disk once and reusing it for
        subsequent boxes/edits on the same image (see _enter_image)."""
        if self._current_cv_image is None and 0 <= self.current_index < len(self.image_list):
            full_path = os.path.join(self.image_dir, self.image_list[self.current_index])
            self._current_cv_image = imread(full_path)
        return self._current_cv_image

    def _recognize_box(self, index: int, *, show_errors: bool, push_undo: bool) -> bool:
        """Re-run OCR recognition on `index`'s current crop and store the
        result as its transcription/score. Shared by the manual "Re-recognize
        Selected Box" menu action (show_errors=True: pops dialogs, always
        pushes its own undo step) and the automatic re-recognition that fires
        after a box is dragged/resized when auto-recognize is on
        (show_errors=False: quiet status-bar notices only, no extra undo
        step since the drag itself already pushed one)."""
        if not (0 <= index < len(self.current_shapes)):
            return False
        if not self.engine.is_available():
            self._notify_ocr_unavailable(show_errors)
            return False
        try:
            img = self._get_current_cv_image()
            if img is None:
                raise RuntimeError("Could not re-read the source image from disk.")
            crop = export_utils.get_rotate_crop_image(img, self.current_shapes[index].points)
            text, score = self.engine.recognize_only(crop)
        except Exception as e:
            if show_errors:
                QMessageBox.warning(self, "Re-recognition failed", str(e))
            else:
                self.statusBar().showMessage(f"Auto re-recognition failed: {e}", 5000)
            return False

        if push_undo:
            self.push_undo()
        self.current_shapes[index].transcription = text
        self.current_shapes[index].score = score
        self._rebuild_table()
        self.table.selectRow(index)
        self.mark_dirty()
        return True

    def _notify_ocr_unavailable(self, show_errors: bool) -> None:
        message = ("Install paddleocr to use re-recognition:\n"
                   "    pip install paddlepaddle paddleocr")
        if show_errors:
            QMessageBox.warning(self, "PaddleOCR not available", message)
        elif not self._auto_recognize_warned:
            self._auto_recognize_warned = True
            self.statusBar().showMessage(
                "Auto re-recognize is on, but paddleocr isn't installed, so "
                "edited boxes won't get new text automatically.", 8000)

    # ==================================================================
    # dataset export
    # ==================================================================
    def _labels_snapshot(self) -> dict:
        """Merge on-disk labels with the (possibly unsaved) current image
        so export always reflects what's on screen."""
        if self.label_store is None:
            return {}
        if self.dirty:
            self.save_current_image_labels()
        return dict(self.label_store.data)

    def export_recognition_dataset(self) -> None:
        if not self._require_folder():
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder for recognition dataset")
        if not out_dir:
            return
        stats = export_utils.export_recognition_dataset(self.image_dir, self._labels_snapshot(), out_dir)
        QMessageBox.information(
            self, "Export complete",
            f"Wrote {stats['total_boxes']} crops "
            f"({stats['train']} train / {stats['val']} val) to:\n{out_dir}"
            + (f"\n\n{stats['images_skipped']} image(s) could not be read and were skipped."
               if stats.get("images_skipped") else "")
        )

    def export_detection_split(self) -> None:
        if not self._require_folder():
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder for Label_train/val.txt")
        if not out_dir:
            return
        stats = export_utils.split_detection_labels(self._labels_snapshot(), out_dir)
        QMessageBox.information(
            self, "Export complete",
            f"Wrote {stats['total_images']} labeled images "
            f"({stats['train']} train / {stats['val']} val) to:\n{out_dir}"
        )

    def build_char_dict(self) -> None:
        if not self._require_folder():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save character dictionary", "custom_dict.txt")
        if not path:
            return
        n = export_utils.build_char_dict(self._labels_snapshot(), path)
        QMessageBox.information(self, "Dictionary built", f"Wrote {n} characters to:\n{path}")

    def _require_folder(self) -> bool:
        if not self.image_dir or self.label_store is None:
            QMessageBox.information(self, "No folder open", "Open an image folder first.")
            return False
        return True

    # ==================================================================
    # misc
    # ==================================================================
    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self, "Keyboard shortcuts",
            "Ctrl+O  Open image folder\n"
            "Ctrl+S  Save current image's boxes\n"
            "PgUp / PgDown  Previous / next image\n"
            "Ctrl+B  Draw box (click-drag)\n"
            "Ctrl+Shift+B  Draw quad (click 4 corners)\n"
            "Delete / Backspace  Delete the selected box (canvas focused)\n"
            "Ctrl+Z / Ctrl+Y  Undo / redo box position, creation, deletion\n"
            "Mouse wheel  Zoom\n"
            "Middle-drag  Pan\n"
            "Ctrl+0  Fit image to window\n"
            "Esc  Cancel drawing / deselect"
        )
