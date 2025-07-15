import sys
import random
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
        self.setWindowTitle("Photon CA — Directional Emission & Reset")
        self.resize(1400, 800)

        # Total number of cells
        N = GRID_SIZE**3

        # State arrays
        self.electric = np.zeros(N, dtype=int)
        self.magnetic = np.zeros(N, dtype=int)
        self.direction = np.zeros((N, 3), dtype=int)

        # History for plots
        self.e_history = []
        self.m_history = []

        # Precompute point coordinates and cloud
        half = GRID_SIZE // 2
        coords = np.indices((GRID_SIZE,)*3).reshape(3, -1).T
        pts = (coords - half) * SPACING
        self.cloud = pv.PolyData(pts)
        self.center_idx = np.ravel_multi_index((half,)*3, (GRID_SIZE,)*3)

        # Layouts
        main_layout = QtWidgets.QHBoxLayout()
        views_layout = QtWidgets.QVBoxLayout()
        controls_layout = QtWidgets.QVBoxLayout()

        # 3D viewers
        self.e3d = QtInteractor()
        self.m3d = QtInteractor()
        self.o3d = QtInteractor()
        self.e3d.enable_point_picking(self.inject_electric, show_message=False)
        views_layout.addWidget(self.e3d.interactor)
        views_layout.addWidget(self.m3d.interactor)
        views_layout.addWidget(self.o3d.interactor)

        # Oscilloscope plots
        self.fig_e, self.ax_e, self.canv_e = self.make_wave("Electric (E)", "blue")
        self.fig_m, self.ax_m, self.canv_m = self.make_wave("Magnetic (B)", "red")
        self.fig_o, self.ax_o, self.canv_o = self.make_wave("Overlay", None)
        controls_layout.addWidget(self.canv_e)
        controls_layout.addWidget(self.canv_m)
        controls_layout.addWidget(self.canv_o)

        # Injection via UI
        inj_layout = QtWidgets.QHBoxLayout()
        self.spin_x = QtWidgets.QSpinBox(); self.spin_x.setRange(-half, half)
        self.spin_y = QtWidgets.QSpinBox(); self.spin_y.setRange(-half, half)
        self.spin_z = QtWidgets.QSpinBox(); self.spin_z.setRange(-half, half)
        inj_btn = QtWidgets.QPushButton("Inject (UI)")
        inj_btn.clicked.connect(self.inject_ui)
        for lbl, sp in zip(("X","Y","Z"), (self.spin_x, self.spin_y, self.spin_z)):
            inj_layout.addWidget(QtWidgets.QLabel(lbl))
            inj_layout.addWidget(sp)
        inj_layout.addWidget(inj_btn)
        controls_layout.addLayout(inj_layout)

        # Step button
        step_btn = QtWidgets.QPushButton("Next Frame")
        step_btn.clicked.connect(self.step)
        controls_layout.addWidget(step_btn)

        # Debug values button
        debug_btn = QtWidgets.QPushButton("Debug Values")
        debug_btn.clicked.connect(self.debug_values)
        controls_layout.addWidget(debug_btn)

        # Reset button
        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        controls_layout.addWidget(reset_btn)

        # Assemble window
        container = QtWidgets.QWidget()
        main_layout.addLayout(views_layout, 3)
        main_layout.addLayout(controls_layout, 2)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initial draw
        self.update_views()
        self.update_plots()

    def inject_electric(self, pick_point):
        idx = np.argmin(np.linalg.norm(self.cloud.points - pick_point, axis=1))
        self._reset_fields()
        self.electric[idx] = 1
        self.direction[idx] = random.choice(
            [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        )
        self.update_views()
        self.update_plots()

    def inject_ui(self):
        half = GRID_SIZE // 2
        x = self.spin_x.value() + half
        y = self.spin_y.value() + half
        z = self.spin_z.value() + half
        idx = np.ravel_multi_index((x,y,z), (GRID_SIZE,)*3)
        self._reset_fields()
        self.electric[idx] = 1
        self.direction[idx] = random.choice(
            [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        )
        self.update_views()
        self.update_plots()

    def step(self):
        n = GRID_SIZE
        coords = np.indices((n,n,n)).reshape(3,-1).T
        newE = np.zeros_like(self.electric)
        newB = np.zeros_like(self.magnetic)
        newDir = np.zeros_like(self.direction)

        occupied = np.nonzero(self.electric | self.magnetic)[0]
        for idx in occupied:
            x,y,z = coords[idx]
            dx,dy,dz = self.direction[idx]
            xt,yt,zt = (x+dx)%n, (y+dy)%n, (z+dz)%n
            tidx = np.ravel_multi_index((xt,yt,zt),(n,n,n))
            if self.electric[idx]:
                newB[tidx] = 1
            else:
                newE[tidx] = 1
            newDir[tidx] = (dx,dy,dz)

        self.electric[:] = newE
        self.magnetic[:] = newB
        self.direction[:] = newDir
        self.update_views()
        self.update_plots()

    def reset(self):
        """Reset all fields, directions, and history."""
        self._reset_fields()
        self.e_history.clear()
        self.m_history.clear()
        self.update_views()
        self.update_plots()

    def _reset_fields(self):
        """Helper to clear bits and directions."""
        self.electric[:] = 0
        self.magnetic[:] = 0
        self.direction[:] = 0

    def update_views(self):
        grey_cloud = pv.PolyData(self.cloud.points)
        # Electric view
        self.e3d.clear()
        self.e3d.add_mesh(grey_cloud, color="lightgray",
                          render_points_as_spheres=True, point_size=8)
        e_mask = self.electric.astype(bool)
        if e_mask.any():
            self.e3d.add_mesh(pv.PolyData(self.cloud.points[e_mask]),
                              color="blue", render_points_as_spheres=True, point_size=8)
        self.e3d.render()
        # Magnetic view
        self.m3d.clear()
        self.m3d.add_mesh(grey_cloud, color="lightgray",
                          render_points_as_spheres=True, point_size=8)
        m_mask = self.magnetic.astype(bool)
        if m_mask.any():
            self.m3d.add_mesh(pv.PolyData(self.cloud.points[m_mask]),
                              color="red", render_points_as_spheres=True, point_size=8)
        self.m3d.render()
        # Overlay view
        self.o3d.clear()
        self.o3d.add_mesh(grey_cloud, color="lightgray",
                          render_points_as_spheres=True, point_size=8)
        if e_mask.any():
            self.o3d.add_mesh(pv.PolyData(self.cloud.points[e_mask]),
                              color="blue", render_points_as_spheres=True, point_size=8)
        if m_mask.any():
            self.o3d.add_mesh(pv.PolyData(self.cloud.points[m_mask]),
                              color="red", render_points_as_spheres=True, point_size=8)
        self.o3d.render()

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
        self.ax_o.plot(t, self.m_history, color="red", label="B")
        self.ax_o.legend(loc='upper right')
        self.canv_o.draw()

    def debug_values(self):
        half = GRID_SIZE // 2
        coords = np.indices((GRID_SIZE,)*3).reshape(3,-1).T - half
        e_idx = np.nonzero(self.electric)[0]
        m_idx = np.nonzero(self.magnetic)[0]
        e_pos = tuple(coords[e_idx[0]]) if e_idx.size else None
        m_pos = tuple(coords[m_idx[0]]) if m_idx.size else None
        QtWidgets.QMessageBox.information(
            self, "Debug Values",
            f"Electric at: {e_pos}\nMagnetic at: {m_pos}"
        )

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

