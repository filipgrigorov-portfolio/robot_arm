import numpy as np
from utils import degrees2rad

EPS = 1e-2

class NewtonSolver2D3L:
    def __init__(self, thetas, links):
        # 3,1
        self.thetas = thetas
        # 3,1
        self.links = links

        self.J = self.compute_Jacobian(thetas, links)

        return self

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

    def solve(self, target, trajectory, thetas, links):
        # dth = J.inv*dp
        jnts = self._FK(thetas, links)
        trajectory.append(jnts)
        dp = jnts[-1] - target
        while np.abs(np.linalg.norm(dp)) < EPS:
            # Note: Can use np.linalg.pinv(self.J)
            JInv = self._pinv()
            dth = JInv.dot(dp)
            thetas += dth
            jnts = self._FK(thetas, links)
            trajectory.append(jnts)

    def _FK(self, thetas, links):
        p0 = links[0]
        p1 = p0 + np.array([ 
            links[0] * np.sin(thetas[0]), 
            links[0] * np.cos(thetas[0]) 
        ])
        p2 = p1 + np.array([ 
            links[1] * np.sin(thetas[0] + thetas[1]), 
            links[1] * np.cos(thetas[0] + thetas[1]) 
        ])
        p3 = p1 + p2 + np.array([ 
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
    thetas = np.array([90.0, 90.0, 90.0]).astype(np.float32)
    thetas = degrees2rad(thetas)
    print(NewtonSolver2D3L(thetas, links).compute_Jacobian(thetas, links))
