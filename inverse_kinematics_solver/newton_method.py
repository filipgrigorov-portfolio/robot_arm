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

        print(self.J)

        return self        

    '''
    def compute_Jacobian(self, thetas, links):
        # len(links) x len(jnts)
        J = np.zeros(shape=(len(links), len(thetas))).astype(np.float32)
        indices = np.array(np.linspace(0, len(thetas) - 1, len(thetas))).astype(int)
        for row in range(len(links)):
            func = np.cos if row % 2 == 0 else -np.sin
            for col in range(len(thetas)):
                J[row][col] = (links[indices[col :]] * func(thetas[indices[col :]])).sum()
        return J
    '''

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
        while np.abs(np.linalg.norm(jnts[-1] - target)) < EPS:
            # compute svd of J
            
            # deduce pseudo inverse from svd
            # compute the theta deltas
            # apply theta deltas to thetas
            # compute the jnts coords with the new thetas
            jnts = self._FK(thetas, links)
            trajectory.append(jnts)

    def _FK(self, thetas, links):
        p0 = links[0]
        p1 = p0 + np.array([ links[0] * np.sin(thetas[0]), links[0] * np.cos(thetas[0]) ])
        p2 = p1 + np.array([ links[1] * np.sin(thetas[0] + thetas[1]), links[1] * np.cos(thetas[0] + thetas[1]) ])
        p3 = p1 + p2 + np.array([ links[2] * np.sin(thetas[0] + thetas[1] + thetas[2]), links[2] * np.cos(thetas[0] + thetas[1] + thetas[2]) ])
        return np.array([ p0, p1, p2, p3 ])

if __name__ == '__main__':
    links = np.array([3, 3, 3]).astype(np.float32)
    thetas = np.array([90.0, 90.0, 90.0]).astype(np.float32)
    thetas = degrees2rad(thetas)
    print(NewtonSolver2D3L(thetas, links).compute_Jacobian(thetas, links))
