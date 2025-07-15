import sys
import numpy as np
from PyQt5 import QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

GRID_SIZE = 7
SPACING = 1.0

class EMFieldApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CA EM Toggle — Explicit Color Layers")
        self.resize(1400, 800)

        # State arrays
        N = GRID_SIZE**3
        self.electric = np.zeros(N, dtype=int)
        self.magnetic = np.zeros(N, dtype=int)

        # Histories
        self.e_history = []
        self.m_history = []

        # Center index
        half = GRID_SIZE // 2
        self.center_idx = np.ravel_multi_index((half,)*3, (GRID_SIZE,)*3)

        # Build all point coordinates
        coords = np.indices((GRID_SIZE,)*3).reshape(3, -1).T
        pts = (coords - half) * SPACING
        self.all_pts = pts

        # Layouts
        main_l = QtWidgets.QHBoxLayout()
        views_l = QtWidgets.QVBoxLayout()
        ctrls_l = QtWidgets.QVBoxLayout()

        # Electric view
        self.e3d = QtInteractor()
        views_l.addWidget(self.e3d.interactor)

        # Magnetic view
        self.m3d = QtInteractor()
        views_l.addWidget(self.m3d.interactor)

        # Overlay view
        self.o3d = QtInteractor()
        views_l.addWidget(self.o3d.interactor)

        # Inject by clicking in E view
        self.e3d.enable_point_picking(self.inject_electric, show_message=False)

        # Oscilloscopes
        self.fig_e, self.ax_e, self.canv_e = self.make_wave("Electric (E)", "blue")
        self.fig_m, self.ax_m, self.canv_m = self.make_wave("Magnetic (B)", "red")
        self.fig_o, self.ax_o, self.canv_o = self.make_wave("Overlay", None)
        for c in (self.canv_e, self.canv_m, self.canv_o):
            ctrls_l.addWidget(c)

        # Controls: UI inject, step, debug
        half = GRID_SIZE // 2
        inj_l = QtWidgets.QHBoxLayout()
        self.spin_x = QtWidgets.QSpinBox(); self.spin_x.setRange(-half, half)
        self.spin_y = QtWidgets.QSpinBox(); self.spin_y.setRange(-half, half)
        self.spin_z = QtWidgets.QSpinBox(); self.spin_z.setRange(-half, half)
        inj_btn = QtWidgets.QPushButton("Inject (UI)")
        inj_btn.clicked.connect(self.inject_ui)
        for lbl, sp in zip(("X","Y","Z"), (self.spin_x, self.spin_y, self.spin_z)):
            inj_l.addWidget(QtWidgets.QLabel(lbl)); inj_l.addWidget(sp)
        inj_l.addWidget(inj_btn)
        ctrls_l.addLayout(inj_l)

        step_btn = QtWidgets.QPushButton("Next Frame")
        step_btn.clicked.connect(self.step)
        ctrls_l.addWidget(step_btn)

        debug_btn = QtWidgets.QPushButton("Debug Values")
        debug_btn.clicked.connect(self.debug_values)
        ctrls_l.addWidget(debug_btn)

        # Assemble
        container = QtWidgets.QWidget()
        main_l.addLayout(views_l, 3)
        main_l.addLayout(ctrls_l, 2)
        container.setLayout(main_l)
        self.setCentralWidget(container)

        # Initial draw
        self.update_views()
        self.update_plots()

    def inject_electric(self, pick_point):
        idx = np.argmin(np.linalg.norm(self.all_pts - pick_point, axis=1))
        self.electric[:] = 0
        self.magnetic[:] = 0
        self.electric[idx] = 1
        self.update_views()
        self.update_plots()

    def inject_ui(self):
        half = GRID_SIZE // 2
        x = self.spin_x.value() + half
        y = self.spin_y.value() + half
        z = self.spin_z.value() + half
        idx = np.ravel_multi_index((x, y, z), (GRID_SIZE,)*3)
        self.electric[:] = 0
        self.magnetic[:] = 0
        self.electric[idx] = 1
        self.update_views()
        self.update_plots()

    def step(self):
        new_e = self.magnetic.copy()
        new_m = self.electric.copy()
        self.electric[:] = 0
        self.magnetic[:] = 0
        self.electric[:] = new_e
        self.magnetic[:] = new_m
        self.update_views()
        self.update_plots()

    def update_views(self):
        # Create grey background of all points
        grey_cloud = pv.PolyData(self.all_pts)

        # Electric view: show grey + blue
        self.e3d.clear()
        self.e3d.add_mesh(grey_cloud, color="lightgray", render_points_as_spheres=True, point_size=8)
        e_mask = self.electric.astype(bool)
        if e_mask.any():
            pts_e = pv.PolyData(self.all_pts[e_mask])
            self.e3d.add_mesh(pts_e, color="blue", render_points_as_spheres=True, point_size=8)
        self.e3d.render()
        QtWidgets.QApplication.processEvents()

        # Magnetic view: show grey + red
        self.m3d.clear()
        self.m3d.add_mesh(grey_cloud, color="lightgray", render_points_as_spheres=True, point_size=8)
        m_mask = self.magnetic.astype(bool)
        if m_mask.any():
            pts_m = pv.PolyData(self.all_pts[m_mask])
            self.m3d.add_mesh(pts_m, color="red", render_points_as_spheres=True, point_size=8)
        self.m3d.render()
        QtWidgets.QApplication.processEvents()

        # Overlay: show grey + blue + red
        self.o3d.clear()
        self.o3d.add_mesh(grey_cloud, color="lightgray", render_points_as_spheres=True, point_size=8)
        if e_mask.any():
            pts_e = pv.PolyData(self.all_pts[e_mask])
            self.o3d.add_mesh(pts_e, color="blue", render_points_as_spheres=True, point_size=8)
        if m_mask.any():
            pts_m = pv.PolyData(self.all_pts[m_mask])
            self.o3d.add_mesh(pts_m, color="red", render_points_as_spheres=True, point_size=8)
        self.o3d.render()
        QtWidgets.QApplication.processEvents()

    def update_plots(self):
        self.e_history.append(int(self.electric[self.center_idx]))
        self.m_history.append(int(self.magnetic[self.center_idx]))
        t = np.arange(len(self.e_history))
        # Electric
        self.ax_e.clear()
        self.ax_e.plot(t, self.e_history, color="blue")
        self.canv_e.draw()
        # Magnetic
        self.ax_m.clear()
        self.ax_m.plot(t, self.m_history, color="red")
        self.canv_m.draw()
        # Overlay
        self.ax_o.clear()
        self.ax_o.plot(t, self.e_history, color="blue", label="E")
        self.ax_o.plot(t, self.m_history, color="red", label="M")
        self.ax_o.legend(loc='upper right')
        self.canv_o.draw()

    def debug_values(self):
        coords = (self.all_pts + GRID_SIZE//2) / SPACING - GRID_SIZE//2
        e_idx = np.nonzero(self.electric)[0]
        m_idx = np.nonzero(self.magnetic)[0]
        e_pos = tuple(coords[e_idx[0]].astype(int)) if e_idx.size else None
        m_pos = tuple(coords[m_idx[0]].astype(int)) if m_idx.size else None
        QtWidgets.QMessageBox.information(self, "Debug Values",
                                          f"Electric active at: {e_pos}\nMagnetic active at: {m_pos}")

    def make_wave(self, title, color):
        fig = Figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlim(0,100)
        ax.set_ylim(0,1.5)
        ax.grid(True)
        if color:
            ax.plot([], [], color=color, label=title.split()[0])
            ax.legend(loc='upper right')
        return fig, ax, FigureCanvas(fig)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EMFieldApp()
    win.show()
    sys.exit(app.exec_())
