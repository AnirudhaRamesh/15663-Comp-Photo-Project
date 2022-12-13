import numpy as np
import matplotlib.pyplot as plt
import pickle

video = 'youtubebridge'
modal_data_file = 'data/%s/modal_observations.pkl' % video
with open(modal_data_file, 'rb') as fp:
    modal_data = pickle.load(fp)

mdx = modal_data['modes_dx']
mdy = modal_data['modes_dy']
freqs = modal_data['freqs']

alpha, beta = 1,1
eps = lambda wi : 1/2 * (alpha / wi + beta * wi) 

def calc_displacement_fields(Q, mdx, mdy) : 

    D_x = np.sum(np.real(Q * mdx),0)
    D_y = np.sum(np.real(Q * mdy),0)

    return D_x, D_y
        
 
def calc_modal_coordinates_pixelwise(mdx, mdy) : 
    modal_coordinates = np.zeros(mdx.shape)
    # speed up later
    for x in range(mdx.shape[1]) : 
        for y in range(mdx.shape[2]) : 
            phi = np.vstack((mdx[:,x,y], mdy[:,x,y]))
            pos = np.array([[x],[y]])
            phi_pinv = np.linalg.pinv(phi) 
            modal_coordinates[:, x, y] = (phi_pinv @ pos).squeeze() 
    return modal_coordinates

def calc_modal_coordinates_matrixwise(mdx, mdy) :
    modal_coordinates = np.zeros(mdx.shape)
    phi = np.stack((mdx, mdy))
    phi_pinv = np.linalg.pinv(phi) # this doesnt work 
    # phi.T @ phi doesn't work either which is required to calc pinv by hand
    colnum, rownum  = np.meshgrid(np.arange(mdx.shape[1]), np.arange(mdx.shape[2]))
    X = np.stack((rownum, colnum))
    modal_coordinates = phi_pinv @ X
    return modal_coordinates
    
def calc_fi(mi_dx, mi_dy, dir = np.array([1,1]), alpha=20) :
    phi_p = np.array([mi_dx, mi_dy]) 
    f_i = np.abs(np.dot(dir, phi_p)) * alpha
    return f_i

def get_next_state_pos(Qi, wi, h, fi, mass_i) : 

    y = np.zeros((2,1))
    y[0] = np.real(Qi)
    y[1] = -np.imag(Qi) * wi
    
    a = np.array([[1,h],[-wi**2 * h, 1- 2 * eps(wi) * wi * h]])
    b = np.array([0, h/mass_i])

    y_new = np.dot(a,y) + b * fi

    mi_d_new = y_new[0] - np.array([0+1j]) * y_new[1] / wi# qi_new

    return mi_d_new[0]

def run_simulation_per_pixel(Qi, mi_dx, mi_dy, wi, h=0.01, dir=np.array([1,1]), mass_i=1, timesteps=1) : 

    for ts in range(timesteps) :
        fi = calc_fi(mi_dx, mi_dy, dir)
        Qi = get_next_state_pos(Qi, wi, h, fi, mass_i)

    return Qi

def run_simulation_over_all_pixels_per_mode(Q, m_dx, m_dy, w, timesteps=3) :
    
    # new_m_dx = np.zeros(m_dx.shape, dtype=complex)
    # new_m_dy = np.zeros(m_dy.shape, dtype=complex)
    new_Q = np.zeros(Q.shape, dtype=complex)
    for x in range(m_dx.shape[0]) : 
        for y in range(m_dx.shape[1]) :
            new_Q[x,y] = run_simulation_per_pixel(Q[x,y], m_dx[x,y], m_dy[x,y], w, timesteps=timesteps)
    
    return new_Q

Q = calc_modal_coordinates_pixelwise(mdx, mdy) 
new_Q = np.zeros(Q.shape, dtype=complex)
new_Q[0] = run_simulation_over_all_pixels_per_mode(Q[0].copy(), mdx[0].copy(), mdy[0].copy(), freqs[0])
new_Q[1] = run_simulation_over_all_pixels_per_mode(Q[1].copy(), mdx[1].copy(), mdy[1].copy(), freqs[1])
new_Q[2] = run_simulation_over_all_pixels_per_mode(Q[2].copy(), mdx[2].copy(), mdy[2].copy(), freqs[2])
new_Q[3] = run_simulation_over_all_pixels_per_mode(Q[3].copy(), mdx[3].copy(), mdy[3].copy(), freqs[3])

D_x, D_y = calc_displacement_fields(new_Q, mdx, mdy)

# new_m_dx, new_m_dy = run_simulation_over_all_pixels_per_mode(mdx[0].copy(), mdy[0].copy(), freqs[0])

plt.imshow(np.real(new_m_dx))
plt.savefig('new_dx_0_real.png')
plt.clf()
plt.cla()

plt.imshow(np.imag(new_m_dx))
plt.savefig('new_dx_0_imag.png')
plt.clf()
plt.cla()

plt.imshow(mdx[0])
plt.savefig('org_dx_0.png')
plt.clf()
plt.cla()

plt.imshow(np.real(new_m_dy))
plt.savefig('new_dy_0_real.png')
plt.clf()
plt.cla()

plt.imshow(np.imag(new_m_dy))
plt.savefig('new_dy_0_imag.png')
plt.clf()
plt.cla()

plt.imshow(mdy[0])
plt.savefig('org_dy_0.png')
plt.clf()
plt.cla()