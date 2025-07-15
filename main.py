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
        self.setWindowTitle("Photon CA — Directional Emission & Propagation")
        self.resize(1400, 800)

        # Total number of cells
        N = GRID_SIZE**3

        # State arrays: electric bit, magnetic bit, and direction vector (dx, dy, dz)
        self.electric = np.zeros(N, dtype=int)
        self.magnetic = np.zeros(N, dtype=int)
        self.direction = np.zeros((N, 3), dtype=int)

        # History for oscilloscope plots
        self.e_history = []
        self.m_history = []

        # Precompute point coordinates
        half = GRID_SIZE // 2
        coords = np.indices((GRID_SIZE,)*3).reshape(3, -1).T
        pts = (coords - half) * SPACING
        self.cloud = pv.PolyData(pts)

        # Center index for history plots
        self.center_idx = np.ravel_multi_index((half,)*3, (GRID_SIZE,)*3)

        # Build UI layouts
        main_layout = QtWidgets.QHBoxLayout()
        views_layout = QtWidgets.QVBoxLayout()
        controls_layout = QtWidgets.QVBoxLayout()

        # 3D viewers
        self.e3d = QtInteractor()
        self.m3d = QtInteractor()
        self.o3d = QtInteractor()

        # Enable point-picking for injection in electric view
        self.e3d.enable_point_picking(self.inject_electric, show_message=False)

        # Add viewers to layout
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

        # Injection by UI controls
        inj_layout = QtWidgets.QHBoxLayout()
        self.spin_x = QtWidgets.QSpinBox(); self.spin_x.setRange(-half, half)
        self.spin_y = QtWidgets.QSpinBox(); self.spin_y.setRange(-half, half)
        self.spin_z = QtWidgets.QSpinBox(); self.spin_z.setRange(-half, half)
        inj_btn = QtWidgets.QPushButton("Inject (UI)")
        inj_btn.clicked.connect(self.inject_ui)
        for label, spin in zip(("X","Y","Z"), (self.spin_x, self.spin_y, self.spin_z)):
            inj_layout.addWidget(QtWidgets.QLabel(label)); inj_layout.addWidget(spin)
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

        # Assemble main window
        container = QtWidgets.QWidget()
        main_layout.addLayout(views_layout, 3)
        main_layout.addLayout(controls_layout, 2)
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initial render and plot
        self.update_views()
        self.update_plots()

    def inject_electric(self, pick_point):
        """Inject a single electric bit at picked location with random direction."""
        # Find nearest cell index
        idx = np.argmin(np.linalg.norm(self.cloud.points - pick_point, axis=1))
        # Reset state
        self.electric[:] = 0
        self.magnetic[:] = 0
        # Set electric bit
        self.electric[idx] = 1
        # Choose random cardinal direction for the photon
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.direction[idx] = random.choice(offsets)
        # Refresh views and plots
        self.update_views()
        self.update_plots()

    def inject_ui(self):
        """Inject via UI spinbox coordinates."""
        half = GRID_SIZE // 2
        x = self.spin_x.value() + half
        y = self.spin_y.value() + half
        z = self.spin_z.value() + half
        idx = np.ravel_multi_index((x, y, z), (GRID_SIZE,)*3)
        self.electric[:] = 0
        self.magnetic[:] = 0
        self.electric[idx] = 1
        offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        self.direction[idx] = random.choice(offsets)
        self.update_views()
        self.update_plots()

    def step(self):
        """Move the single photon bit along its direction, flipping E↔B."""
        n = GRID_SIZE
        coords = np.indices((n, n, n)).reshape(3, -1).T

        newE = np.zeros_like(self.electric)
        newB = np.zeros_like(self.magnetic)
        newDir = np.zeros_like(self.direction)

        occupied = np.nonzero(self.electric | self.magnetic)[0]
        for idx in occupied:
            x, y, z = coords[idx]
            dx, dy, dz = self.direction[idx]
            # Compute target with wrap-around
            xt, yt, zt = (x + dx) % n, (y + dy) % n, (z + dz) % n
            tidx = np.ravel_multi_index((xt, yt, zt), (n, n, n))
            # Flip field bit at target
            if self.electric[idx]:
                newB[tidx] = 1
            else:
                newE[tidx] = 1
            # Carry direction forward
            newDir[tidx] = (dx, dy, dz)

        self.electric[:] = newE
        self.magnetic[:] = newB
        self.direction[:] = newDir

        self.update_views()
        self.update_plots()

    def update_views(self):
        """Render the point mesh for E, B, and overlay views."""
        # Base gray cloud
        grey_cloud = pv.PolyData(self.cloud.points)

        # Electric view: gray + blue points
        self.e3d.clear()
        self.e3d.add_mesh(grey_cloud, color="lightgray",
                          render_points_as_spheres=True, point_size=8)
        e_mask = self.electric.astype(bool)
        if e_mask.any():
            pts_e = pv.PolyData(self.cloud.points[e_mask])
            self.e3d.add_mesh(pts_e, color="blue",
                              render_points_as_spheres=True, point_size=8)
        self.e3d.render()

        # Magnetic view: gray + red points
        self.m3d.clear()
        self.m3d.add_mesh(grey_cloud, color="lightgray",
                          render_points_as_spheres=True, point_size=8)
        m_mask = self.magnetic.astype(bool)
        if m_mask.any():
            pts_m = pv.PolyData(self.cloud.points[m_mask])
            self.m3d.add_mesh(pts_m, color="red",
                              render_points_as_spheres=True, point_size=8)
        self.m3d.render()

        # Overlay view: gray + blue + red
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
        """Update oscilloscope plots for E, B, and overlay."""
        self.e_history.append(int(self.electric[self.center_idx]))
        self.m_history.append(int(self.magnetic[self.center_idx]))
        t = np.arange(len(self.e_history))

        # E plot
        self.ax_e.clear()
        self.ax_e.plot(t, self.e_history, color="blue")
        self.canv_e.draw()

        # B plot
        self.ax_m.clear()
        self.ax_m.plot(t, self.m_history, color="red")
        self.canv_m.draw()

        # Overlay plot
        self.ax_o.clear()
        self.ax_o.plot(t, self.e_history, color="blue", label="E")
        self.ax_o.plot(t, self.m_history, color="red", label="B")
        self.ax_o.legend(loc='upper right')
        self.canv_o.draw()

    def debug_values(self):
        """Show current active E/B coordinates."""
        half = GRID_SIZE // 2
        coords = np.indices((GRID_SIZE,)*3).reshape(3, -1).T - half
        e_idx = np.nonzero(self.electric)[0]
        m_idx = np.nonzero(self.magnetic)[0]
        e_pos = tuple(coords[e_idx[0]]) if e_idx.size else None
        m_pos = tuple(coords[m_idx[0]]) if m_idx.size else None
        QtWidgets.QMessageBox.information(
            self, "Debug Values",
            f"Electric at: {e_pos}\nMagnetic at: {m_pos}"
        )

    def make_wave(self, title, color):
        """Helper to create a matplotlib FigureCanvas for oscilloscope."""
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

