import numpy as np

EPS = 1e-2

def fabrik_solve(target, trajectory, jnts, links):
    # check if the target is out of reach
    dist = np.linalg.norm(jnts[0] - target)
    if links.sum() <= dist:
        print('The robot arm is not within reach')
        for idx in range(0, jnts.shape[0] - 1):
            r = np.linalg.norm(target - jnts[idx])
            prop = links[idx] / r
            jnts[idx + 1] =  (1.0 - prop) * jnts[idx] + prop * target
            trajectory.append(np.copy(jnts))
        return np.array(trajectory)
    
    
    print('The robot arm is within reach')
    base = np.copy(jnts[0]) # Note: we do not need a reference
    dist = np.linalg.norm(jnts[-1] - target)
    while dist > EPS:
        # forward
        jnts[-1] = target
        for idx in range(0, jnts.shape[0] - 1):
            idx = jnts.shape[0] - 2 - idx
            r = np.linalg.norm(jnts[idx + 1] - jnts[idx])
            prop = links[idx] / r
            jnts[idx] = (1.0 - prop) * jnts[idx + 1] + prop * jnts[idx]

        # backward
        jnts[0] = base
        for idx in range(0, jnts.shape[0] - 1):
            r = np.linalg.norm(jnts[idx + 1] - jnts[idx])
            prop = links[idx] / r
            jnts[idx + 1] = (1.0 - prop) * jnts[idx] + prop * jnts[idx + 1]
        dist = np.linalg.norm(jnts[-1] - target)
        print(f'\rdist= {dist}', end='')
        
        # Collect trajectory data
        trajectory.append(np.copy(jnts))
    return np.array(trajectory)
