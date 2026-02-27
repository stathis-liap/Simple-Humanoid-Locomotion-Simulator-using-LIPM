import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LIPMVisualizer:
    def __init__(self, simulator, config, referee=None):
        self.sim = simulator
        self.config = config
        self.h = config["h"]
        self.dt = config["dt"]
        self.referee = referee  
        
        self.history_len = 200
        self.t_data = np.zeros(self.history_len)
        self.p_data = np.zeros(self.history_len)
        self.u_data = np.zeros(self.history_len)
        self.xi_data = np.zeros(self.history_len)
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax_anim = self.fig.add_subplot(2, 1, 1)
        self.ax_plot = self.fig.add_subplot(2, 1, 2)
        self._setup_plots() 

    #helper
    def _setup_plots(self):
        self.ax_anim.set_xlim(-20.0, 20.0)
        self.ax_anim.set_ylim(-0.1, self.h + 0.5)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.grid(True)
        self.ground_line, = self.ax_anim.plot([-20, 20], [0, 0], 'k-', lw=2)
        self.leg_line, = self.ax_anim.plot([], [], 'k-', lw=4, label='Leg')
        self.com_point, = self.ax_anim.plot([], [], 'bo', ms=10, label='COM')
        self.foot_point, = self.ax_anim.plot([], [], 'rs', ms=10, label='Foot')
        self.cp_point, = self.ax_anim.plot([], [], 'gx', ms=8, markeredgewidth=3, label='CP')
        self.ax_anim.legend(loc='upper right')

        self.ax_plot.set_xlim(0, 5)
        self.ax_plot.set_ylim(-1.0, 1.0)
        self.ax_plot.grid(True)
        self.line_p, = self.ax_plot.plot([], [], 'b-', label='Position')
        self.line_u, = self.ax_plot.plot([], [], 'r--', label='Control')
        self.line_xi, = self.ax_plot.plot([], [], 'g:', label='Capture Point')
        self.ax_plot.legend()

    def init_anim(self):
        self.leg_line.set_data([], [])
        self.com_point.set_data([], [])
        self.foot_point.set_data([], [])
        self.cp_point.set_data([], [])
        return self.leg_line, self.com_point, self.foot_point, self.cp_point

    def update(self, frame):
        # Step 
        self.sim.step()
        
        # Check the Referee for GAME OVER
        if self.referee and self.referee.just_fell:
            print(f"-> Referee signaled FALL at t={self.sim.t:.2f}s! Resetting simulation...")
            
            # Reset Physics State
            self.sim.x = np.zeros(2)  # Back to [0, 0]
            self.sim.t = 0.0          # Reset clock
            
            # Reset Visualization Buffers (Clear the history)
            self.t_data[:] = 0
            self.p_data[:] = 0
            self.u_data[:] = 0
            self.xi_data[:] = 0

            # make the leg red and thick for one frame
            self.leg_line.set_color('red')
            self.leg_line.set_linewidth(10)

            return self.leg_line, self.com_point, self.foot_point
        else:
            # Normal color
            self.leg_line.set_color('black')
            self.leg_line.set_linewidth(4)

        # Normal Visualization Logic 
        t = self.sim.t
        p = self.sim.x[0]
        v = self.sim.x[1]
        
        #move the camera for tracking
        self.ax_anim.set_xlim(p - 2.0, p + 2.0)
        
        # compute the (relative) u
        u = self.sim.current_u
        
        # Calculate Capture Point for viz
        omega = np.sqrt(9.81 / self.h)
        xi = p + v / omega

        # Update Animation (Stick Figure)
        self.leg_line.set_data([u, p], [0, self.h])
        self.com_point.set_data([p], [self.h])
        self.foot_point.set_data([u], [0])
        self.cp_point.set_data([xi], [0]) 

        # Update Scrolling Plots
        self.t_data[:-1] = self.t_data[1:]
        self.p_data[:-1] = self.p_data[1:]
        self.u_data[:-1] = self.u_data[1:]
        self.xi_data[:-1] = self.xi_data[1:]
        
        self.t_data[-1] = t
        self.p_data[-1] = p
        self.u_data[-1] = u
        self.xi_data[-1] = xi

        self.line_p.set_data(self.t_data, self.p_data)
        self.line_u.set_data(self.t_data, self.u_data)
        self.line_xi.set_data(self.t_data, self.xi_data)
        
        # Dynamically adjust X-axis to scroll (locked to 0 at start)
        self.ax_plot.set_xlim(max(0, t - 2.0), max(2.0, t + 0.1))

        # Dynamically adjust Y-axis
        # Find the min and max of the data currently in the buffer
        y_min = min(np.min(self.p_data), np.min(self.u_data), np.min(self.xi_data))
        y_max = max(np.max(self.p_data), np.max(self.u_data), np.max(self.xi_data))
        
        # Calculate the middle point
        mid_y = (y_max + y_min) / 2.0
        
        # Ensure the window never zooms in *too* much (keep at least a +/- 1.0 spread)
        # Add a 0.2 margin so lines don't touch the exact top/bottom of the plot
        half_range = max(1.0, (y_max - y_min) / 2.0 + 0.2)
        
        # Apply the new sliding vertical window
        self.ax_plot.set_ylim(mid_y - half_range, mid_y + half_range)

        return self.leg_line, self.com_point, self.foot_point

    def show(self):
        ani = FuncAnimation(self.fig, self.update, init_func=self.init_anim, 
                            frames=None, interval=20, blit=False)
        plt.show()