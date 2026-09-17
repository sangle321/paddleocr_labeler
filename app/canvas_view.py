"""
Interactive canvas for viewing an image and editing quadrilateral
text-box annotations, built on QGraphicsView / QGraphicsScene.

Coordinate convention: the scene uses the same coordinate system as the
underlying image (its raw pixel coordinates). Zooming is done purely
through the view's transform (scale), never by resizing scene content,
so every point stored on a shape is always in original-image pixel
space -- exactly what the label file needs, with no back-and-forth
conversion required when saving.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QKeyEvent, QMouseEvent, QPainter,
                            QPen, QPixmap, QPolygonF, QWheelEvent)
from PySide6.QtWidgets import (QGraphicsPixmapItem, QGraphicsPolygonItem,
                                QGraphicsScene, QGraphicsView)

HANDLE_SIZE = 8   # screen pixels, independent of zoom
HIT_MARGIN = 9    # screen pixels, independent of zoom

NORMAL_COLOR = QColor(0, 200, 83)
DIFFICULT_COLOR = QColor(255, 152, 0)
SELECTED_COLOR = QColor(41, 121, 255)
HANDLE_FILL = QColor(255, 255, 255)


class ShapeItem(QGraphicsPolygonItem):
    """One editable 4-point quadrilateral drawn on the canvas."""

    def __init__(self, points: List[Tuple[float, float]]):
        super().__init__()
        self.points: List[QPointF] = [QPointF(x, y) for x, y in points]
        self.selected = False
        self.difficult = False
        self.setZValue(1)
        self.setPolygon(QPolygonF(self.points))
        self._restyle()

    def move_vertex(self, vertex_index: int, new_point: QPointF) -> None:
        self.points[vertex_index] = QPointF(new_point)
        self.setPolygon(QPolygonF(self.points))

    def translate_points(self, dx: float, dy: float) -> None:
        self.points = [QPointF(p.x() + dx, p.y() + dy) for p in self.points]
        self.setPolygon(QPolygonF(self.points))

    def get_points(self) -> List[Tuple[float, float]]:
        return [(p.x(), p.y()) for p in self.points]

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self._restyle()

    def set_difficult(self, difficult: bool) -> None:
        self.difficult = difficult
        self._restyle()

    def _restyle(self) -> None:
        color = SELECTED_COLOR if self.selected else (
            DIFFICULT_COLOR if self.difficult else NORMAL_COLOR)
        pen = QPen(color, 3 if self.selected else 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(55 if self.selected else 35)
        self.setBrush(QBrush(fill))
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if not self.selected:
            return
        views = self.scene().views() if self.scene() else []
        scale = views[0].transform().m11() if views else 1.0
        scale = scale or 1.0
        r = HANDLE_SIZE / (2 * scale)
        painter.setPen(QPen(QColor(20, 20, 20), 1 / scale))
        painter.setBrush(QBrush(HANDLE_FILL))
        for p in self.points:
            painter.drawRect(QRectF(p.x() - r, p.y() - r, r * 2, r * 2))


class CanvasView(QGraphicsView):
    MODE_EDIT = "edit"
    MODE_RECT = "rect"
    MODE_QUAD = "quad"

    shapesChanged = Signal(int)     # index of shape whose geometry was just committed (drag released)
    shapeCreated = Signal(list)     # [[x,y]x4] -- a new box was just drawn
    selectionChanged = Signal(int)  # index, or -1 for none
    dragStarted = Signal()          # a vertex/whole-shape drag is beginning
    deleteRequested = Signal(int)   # Delete/Backspace pressed with a shape selected
    modeChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setBackgroundBrush(QColor(45, 45, 48))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.shape_items: List[ShapeItem] = []
        self.selected_index: int = -1
        self.mode: str = self.MODE_EDIT

        self._drag_vertex: Optional[Tuple[int, int]] = None
        self._drag_whole: Optional[int] = None
        self._drag_last_scene_pos: Optional[QPointF] = None
        self._rect_start: Optional[QPointF] = None
        self._rect_temp_item = None
        self._quad_points: List[QPointF] = []
        self._panning = False
        self._pan_start = None

    # ------------------------------------------------------------------
    # loading image / shapes
    # ------------------------------------------------------------------
    def load_image(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self.shape_items = []
        self.selected_index = -1
        self.pixmap_item = self._scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)
        self._scene.setSceneRect(0, 0, max(pixmap.width(), 1), max(pixmap.height(), 1))

    def set_shapes(self, shapes) -> None:
        for it in self.shape_items:
            self._scene.removeItem(it)
        self.shape_items = []
        for s in shapes:
            item = ShapeItem(s.points)
            item.set_difficult(s.difficult)
            self._scene.addItem(item)
            self.shape_items.append(item)
        self.selected_index = -1

    def get_shapes_points(self) -> List[List[Tuple[float, float]]]:
        return [it.get_points() for it in self.shape_items]

    def add_shape(self, points, difficult: bool = False, select: bool = True) -> int:
        item = ShapeItem(points)
        item.set_difficult(difficult)
        self._scene.addItem(item)
        self.shape_items.append(item)
        index = len(self.shape_items) - 1
        if select:
            self.select_shape(index)
        return index

    def remove_shape(self, index: int) -> None:
        if not (0 <= index < len(self.shape_items)):
            return
        self._scene.removeItem(self.shape_items[index])
        del self.shape_items[index]
        if self.selected_index == index:
            self.selected_index = -1
        elif self.selected_index > index:
            self.selected_index -= 1

    def select_shape(self, index: int) -> None:
        for i, it in enumerate(self.shape_items):
            it.set_selected(i == index)
        self.selected_index = index
        self.selectionChanged.emit(index)

    def set_shape_difficult(self, index: int, difficult: bool) -> None:
        if 0 <= index < len(self.shape_items):
            self.shape_items[index].set_difficult(difficult)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._quad_points = []
        if self._rect_temp_item is not None:
            self._scene.removeItem(self._rect_temp_item)
            self._rect_temp_item = None
        self.setCursor(Qt.ArrowCursor if mode == self.MODE_EDIT else Qt.CrossCursor)
        self.modeChanged.emit(mode)

    def fit_to_window(self) -> None:
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def zoom_by(self, factor: float) -> None:
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        self.resetTransform()

    # ------------------------------------------------------------------
    # hit testing
    # ------------------------------------------------------------------
    def _current_scale(self) -> float:
        return self.transform().m11() or 1.0

    def _find_vertex_at(self, scene_pos: QPointF) -> Optional[Tuple[int, int]]:
        margin = HIT_MARGIN / self._current_scale()
        best = None
        best_dist = margin
        for si, item in enumerate(self.shape_items):
            for vi, p in enumerate(item.points):
                dist = math.hypot(p.x() - scene_pos.x(), p.y() - scene_pos.y())
                if dist <= best_dist:
                    best_dist = dist
                    best = (si, vi)
        return best

    def _find_shape_at(self, scene_pos: QPointF) -> Optional[int]:
        for si in range(len(self.shape_items) - 1, -1, -1):
            if self.shape_items[si].polygon().containsPoint(scene_pos, Qt.OddEvenFill):
                return si
        return None

    # ------------------------------------------------------------------
    # mouse handling
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        if self.mode == self.MODE_RECT:
            self._rect_start = scene_pos
            return
        if self.mode == self.MODE_QUAD:
            self._quad_points.append(scene_pos)
            if len(self._quad_points) == 4:
                pts = [(p.x(), p.y()) for p in self._quad_points]
                self._quad_points = []
                self.set_mode(self.MODE_EDIT)
                self.shapeCreated.emit(pts)
            return

        # edit mode: vertex drag takes priority over whole-shape drag
        vertex_hit = self._find_vertex_at(scene_pos)
        if vertex_hit is not None:
            self.select_shape(vertex_hit[0])
            self._drag_vertex = vertex_hit
            self.dragStarted.emit()
            return
        shape_hit = self._find_shape_at(scene_pos)
        if shape_hit is not None:
            self.select_shape(shape_hit)
            self._drag_whole = shape_hit
            self._drag_last_scene_pos = scene_pos
            self.dragStarted.emit()
            return
        self.select_shape(-1)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())

        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.mode == self.MODE_RECT and self._rect_start is not None:
            if self._rect_temp_item is not None:
                self._scene.removeItem(self._rect_temp_item)
            rect = QRectF(self._rect_start, scene_pos).normalized()
            pen = QPen(SELECTED_COLOR, 2)
            pen.setCosmetic(True)
            self._rect_temp_item = self._scene.addRect(rect, pen)
            return

        if self._drag_vertex is not None:
            si, vi = self._drag_vertex
            self.shape_items[si].move_vertex(vi, scene_pos)
            return

        if self._drag_whole is not None and self._drag_last_scene_pos is not None:
            delta = scene_pos - self._drag_last_scene_pos
            self._drag_last_scene_pos = scene_pos
            self.shape_items[self._drag_whole].translate_points(delta.x(), delta.y())
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor if self.mode == self.MODE_EDIT else Qt.CrossCursor)
            return

        if self.mode == self.MODE_RECT and self._rect_start is not None:
            scene_pos = self.mapToScene(event.pos())
            if self._rect_temp_item is not None:
                self._scene.removeItem(self._rect_temp_item)
                self._rect_temp_item = None
            rect = QRectF(self._rect_start, scene_pos).normalized()
            self._rect_start = None
            self.set_mode(self.MODE_EDIT)
            if rect.width() > 2 and rect.height() > 2:
                pts = [(rect.left(), rect.top()), (rect.right(), rect.top()),
                       (rect.right(), rect.bottom()), (rect.left(), rect.bottom())]
                self.shapeCreated.emit(pts)
            return

        if self._drag_vertex is not None:
            si, _vi = self._drag_vertex
            self._drag_vertex = None
            self.shapesChanged.emit(si)
            return

        if self._drag_whole is not None:
            si = self._drag_whole
            self._drag_whole = None
            self._drag_last_scene_pos = None
            self.shapesChanged.emit(si)
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.set_mode(self.MODE_EDIT)
            self.select_shape(-1)
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_index >= 0:
            self.deleteRequested.emit(self.selected_index)
        else:
            super().keyPressEvent(event)
