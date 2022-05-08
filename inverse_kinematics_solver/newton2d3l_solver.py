import numpy as np
from utils import degrees2rad

EPS = 1e-1

class NewtonSolver2D3L:
    def __init__(self, thetas, links):
        # 3,1
        self.thetas = thetas
        # 3,1
        self.links = links

        self.J = self.compute_Jacobian(thetas, links)

    def compute_Jacobian(self, thetas, links):
        dpx_dth1 = links[0] * np.cos(thetas[0])
        dpx_dth2 = links[0] * np.cos(thetas[0]) + links[1] * np.cos(thetas[0] + thetas[1])
        dpx_dth3 = links[0] * np.cos(thetas[0]) + links[1] * np.cos(thetas[0] + thetas[1]) \
            + links[2] * np.cos(thetas[0] + thetas[1] + thetas[2])
        dpy_dth1 = -links[0] * np.sin(thetas[0])
        dpy_dth2 = -links[0] * np.sin(thetas[0]) - links[1] * np.sin(thetas[0] + thetas[1])
        dpy_dth3 = -links[0] * np.sin(thetas[0]) - links[1] * np.sin(thetas[0] + thetas[1]) \
            - links[2] * np.sin(thetas[0] + thetas[1] + thetas[2])
        return np.array([
            [dpx_dth1, dpx_dth2, dpx_dth3],
            [dpy_dth1, dpy_dth2, dpy_dth3]
        ]).astype(np.float32)

    def solve(self, target, trajectory, thetas, links, jnts):
        # dth = J.inv*dp
        jnts = self._FK(thetas, links, jnts)
        trajectory.append(jnts)
        dp = jnts[-1] - target
        dist = np.abs(np.linalg.norm(dp))
        while dist > EPS:
            # Note: Can use np.linalg.pinv(self.J)
            JInv = self._pinv()
            dth = JInv.dot(dp)
            thetas += dth
            self.J = self.compute_Jacobian(thetas, links)
            jnts = self._FK(thetas, links, jnts)
            trajectory.append(jnts)
            dp = jnts[-1] - target
            dist = np.abs(np.linalg.norm(dp))
            print(f'\rLoss: {dist}', end='')
        return np.array(trajectory)

    def _FK(self, thetas, links, jnts):
        p0 = jnts[0]
        p1 = p0 + np.array([ 
            links[0] * np.sin(thetas[0]), 
            links[0] * np.cos(thetas[0]) 
        ])
        p2 = p1 + np.array([ 
            links[1] * np.sin(thetas[0] + thetas[1]), 
            links[1] * np.cos(thetas[0] + thetas[1]) 
        ])
        p3 = p2 + np.array([ 
            links[2] * np.sin(thetas[0] + thetas[1] + thetas[2]), 
            links[2] * np.cos(thetas[0] + thetas[1] + thetas[2]) 
        ])
        return np.array([ p0, p1, p2, p3 ])

    def _pinv(self):
        u, s, vh = np.linalg.svd(self.J)
        uInv = u.T
        sInv = np.zeros_like(self.J)
        np.fill_diagonal(sInv, np.where(s > 0, 1.0 / s, s))
        vhInv = vh.T
        return vhInv.dot(sInv.T.dot(uInv))

if __name__ == '__main__':
    links = np.array([3, 3, 3]).astype(np.float32)
    jnts = np.array([[1, 1], [1, 4], [1, 7], [1, 10]]).astype(np.float32)
    thetas = degrees2rad(np.array([0.0, 0.0, 0.0]))

    trajectory = [ np.copy(jnts) ]
    target = np.array([4, 4])
    
    solver = NewtonSolver2D3L(thetas, links)
    trajectory = solver.solve(target, trajectory, thetas, links, jnts)

    print(trajectory)
