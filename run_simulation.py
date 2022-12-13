import numpy as np
import matplotlib.pyplot as plt
import pickle
import vvt
import cv2

video = 'youtubebridge'
modal_data_file = 'data/%s/modal_observations.pkl' % video
with open(modal_data_file, 'rb') as fp:
    modal_data = pickle.load(fp)

mdx = modal_data['modes_dx']
mdy = modal_data['modes_dy']
freqs = modal_data['freqs']

alpha, beta = 1,1
eps = lambda wi : 1/2 * (alpha / wi + beta * wi) 

force_application_x_range =  [210,290]#[200,250]
force_application_y_range =  [300,400]#[100,200]

def save_im_vvt_style(image,title) : 
    vvt.vis.plot_motion_field(image)
    plt.savefig(title)
    plt.clf()

def save_im(image,title) : 
    plt.imshow(image)
    plt.savefig(title)
    plt.clf()

def nonoutlier_mask(val_arr, pct=98):
    thresh = np.percentile(abs(val_arr), pct)
    mask = (abs(val_arr) <= thresh).astype(float)
    return mask

def create_displaced_image(D_x, D_y, ref_frame) : 
    # initialize new image
    shift_dict = {}
    for row in range(D_x.shape[0]) : 
        for col in range(D_x.shape[1]) : 
            new_pixel_row = row + D_x[row, col] 
            new_pixel_col = col + D_y[row, col]
            if new_pixel_row < 0 or new_pixel_row > D_x.shape[0]  : continue
            if new_pixel_col < 0 or new_pixel_col > D_x.shape[1]  : continue
            new_pixel_shift_abs = np.abs(D_x[row, col] ) + np.abs(D_y[row, col])
            if shift_dict.get((new_pixel_row, new_pixel_col)) is None : 
                shift_dict[(new_pixel_row, new_pixel_col)] = [row, col, new_pixel_shift_abs]
            else : 
                [_, _, max_pixel_shift_abs] = shift_dict.get((new_pixel_row, new_pixel_col))
                if max_pixel_shift_abs < new_pixel_shift_abs : # if shift is more on this pixel, use this, else don't change
                    shift_dict[(new_pixel_row, new_pixel_col)] = [row, col, new_pixel_shift_abs]
    new_image = ref_frame.copy()
    for k, v in shift_dict.items() : 
        # k is index in new frame, v is value from ref frame
        new_image[int(k[0]),int(k[1])] = ref_frame[v[0], v[1]]
    return new_image

def read_ref_frame(vfile='data/YoutubeBridge.mp4', start_frame=100) :
    reader = cv2.VideoCapture(vfile)
    frame_no = 0
    while True :
        if frame_no == start_frame : 
            _, im = reader.read()
            break
        _, _ = reader.read() 
        frame_no += 1
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # testing with gray
    return im

def calc_displacement_fields(Qx, Qy, mdx, mdy) : 

    D_x = np.sum(np.real(Qx * mdx),0)
    D_y = np.sum(np.real(Qy * mdy),0)

    return D_x, D_y
        
 
def calc_modal_coordinates_pixelwise(mdx, mdy) : 
    modal_coordinates_qx = np.zeros(mdx.shape)
    modal_coordinates_qy = np.zeros(mdy.shape)
    # speed up later
    for x in range(mdx.shape[1]) : 
        for y in range(mdx.shape[2]) : 
            phi_x =mdx[:,x,y]
            phi_y =mdy[:,x,y]
            modal_coordinates_qx[:, x, y] = x / phi_x
            modal_coordinates_qy[:, x, y] = y / phi_y
    modal_coordinates = np.stack((modal_coordinates_qx, modal_coordinates_qy),0)
    return modal_coordinates

    
def calc_fi(mi_dx, mi_dy, dir, alpha=0.5, cur_x=None, cur_y=None) :
    if cur_x is None and cur_y is None :
        phi_p = np.array([mi_dx, mi_dy]) 
        f_i = np.abs(np.dot(dir, phi_p)) * alpha
        return f_i
    
    if cur_x < force_application_x_range[0] or cur_x > force_application_x_range[1] : 
        alpha = 0
    if cur_y < force_application_y_range[0] or cur_y > force_application_y_range[1] : 
        alpha = 0
    phi_p = np.array([mi_dx, mi_dy]) 
    f_i = np.abs(np.dot(dir, phi_p)) * alpha
    return f_i
 
def get_next_state_pos(Qi, wi, h, fi, mass_i) : 

    y = np.zeros((2,1))
    y[0] = np.real(Qi)
    y[1] = -np.imag(Qi) * wi
    
    a = np.array([[1,h],[-wi**2 * h, 1- 2 * eps(wi) * wi * h]])
    b = np.expand_dims(np.array([0, h/mass_i]),1)
    y_new = np.dot(a,y) + b * fi

    mi_d_new = y_new[0] - np.array([0+1j]) * y_new[1] / wi# qi_new

    return mi_d_new[0]

def run_simulation_per_pixel(Qi, mi_dx, mi_dy, wi, h=0.04, dir=np.array([-1,-1]), mass_i=1, timesteps=1, type_manip=0, cur_x=None, cur_y=None) : 

    for ts in range(timesteps) :
        if type_manip == 0 : 
            fi = calc_fi(mi_dx, mi_dy, dir, cur_x=cur_x, cur_y=cur_y)
            Qi = get_next_state_pos(Qi, wi, h, fi, mass_i)
        elif type_manip == 1 : 
            Qi = calc_fi(mi_dx, mi_dy, dir, cur_x=cur_x, cur_y=cur_y)
            break

    return Qi

def run_simulation_over_all_pixels_per_mode(Q, m_dx, m_dy, w, timesteps=5) :
    
    # new_m_dx = np.zeros(m_dx.shape, dtype=complex)
    # new_m_dy = np.zeros(m_dy.shape, dtype=complex)
    new_Q = np.zeros(Q.shape, dtype=complex)
    for x in range(m_dx.shape[0]) : 
        for y in range(m_dx.shape[1]) :
            new_Q[0,x,y] = run_simulation_per_pixel(Q[0,x,y], m_dx[x,y], m_dy[x,y], w, timesteps=timesteps, cur_x=x, cur_y=y, type_manip=0)
            new_Q[1,x,y] = run_simulation_per_pixel(Q[1,x,y], m_dx[x,y], m_dy[x,y], w, timesteps=timesteps, cur_x=x, cur_y=y, type_manip=0)
    
    return new_Q

Q = calc_modal_coordinates_pixelwise(mdx, mdy) 
Q = np.zeros(Q.shape)
new_Q = np.zeros(Q.shape, dtype=complex)
print(-1)
new_Q[:,0] = run_simulation_over_all_pixels_per_mode(Q[:,0].copy(), mdx[0].copy(), mdy[0].copy(), freqs[0])
print(0)
new_Q[:,1] = run_simulation_over_all_pixels_per_mode(Q[:,1].copy(), mdx[1].copy(), mdy[1].copy(), freqs[1])
print(1)
new_Q[:,2] = run_simulation_over_all_pixels_per_mode(Q[:,2].copy(), mdx[2].copy(), mdy[2].copy(), freqs[2])
print(2)
new_Q[:,3] = run_simulation_over_all_pixels_per_mode(Q[:,3].copy(), mdx[3].copy(), mdy[3].copy(), freqs[3])
print(3)

D_x, D_y = calc_displacement_fields(new_Q[0],new_Q[1], mdx, mdy) 
# D_x, D_y = D_x / 500 , D_y / 500 

org_D_x, org_D_y = calc_displacement_fields(Q[0],Q[1], mdx, mdy)

D_x_nonoutlier_mask = nonoutlier_mask(D_x, 100) # for targetted use 98-99 (depends on window though) 
D_y_nonoutlier_mask = nonoutlier_mask(D_y, 100) # for non-targetted use ~95 maybe. instructions for both.

D_x_nonoutlier = D_x * D_x_nonoutlier_mask
D_y_nonoutlier = D_y * D_y_nonoutlier_mask

from scipy.ndimage import gaussian_filter
D_x_filtered = gaussian_filter(D_x_nonoutlier, 5)
D_y_filtered = gaussian_filter(D_y_nonoutlier, 5)
#D_x_filtered = cv2.GaussianBlur(D_x_nonoutlier, (5,5), 5000)
#D_y_filtered = cv2.GaussianBlur(D_y_nonoutlier, (5,5), 5000)


ref_frame = read_ref_frame()
new_im = create_displaced_image(D_x_filtered, D_y_filtered, ref_frame)

save_im_vvt_style(D_x_filtered, 'D_x_sigma5_maniptype0_targetted_ts5.png')
save_im_vvt_style(D_y_filtered, 'D_y_sigma5_maniptype0_targetted_ts5.png')
save_im(new_im, 'new_im_sigma5_maniptype0_targetted_ts5.png')

D_x_filtered = gaussian_filter(D_x_nonoutlier, 80)
D_y_filtered = gaussian_filter(D_y_nonoutlier, 80)
#D_x_filtered = cv2.GaussianBlur(D_x_nonoutlier, (51,51), 5000)
#D_y_filtered = cv2.GaussianBlur(D_y_nonoutlier, (51,51), 5000)

new_im = create_displaced_image(D_x_filtered, D_y_filtered, ref_frame)

save_im_vvt_style(D_x_filtered, 'D_x_sigma80_maniptype0_targetted_ts5.png')
save_im_vvt_style(D_y_filtered, 'D_y_sigma80_maniptype0_targetted_ts5.png')
save_im(new_im, 'new_im_sigma80_maniptype0_targetted_ts5.png')

D_x_filtered = gaussian_filter(D_x_nonoutlier, 20)
D_y_filtered = gaussian_filter(D_y_nonoutlier, 20)

#D_x_filtered = cv2.GaussianBlur(D_x_nonoutlier, (21,21), 5000)
#D_y_filtered = cv2.GaussianBlur(D_y_nonoutlier, (21,21), 5000)

new_im = create_displaced_image(D_x_filtered, D_y_filtered, ref_frame)

save_im_vvt_style(D_x_filtered, 'D_x_sigma20_maniptype0_targetted_ts5.png')
save_im_vvt_style(D_y_filtered, 'D_y_sigma20_maniptype0_targetted_ts5.png')
save_im(new_im, 'new_im_sigma20_maniptype0_targetted_ts5.png')

# D_x_filtered = gaussian_filter(D_x_nonoutlier, 200)
# D_y_filtered = gaussian_filter(D_y_nonoutlier, 200)

# new_im = create_displaced_image(D_x_filtered, D_y_filtered, ref_frame)

# save_im(new_im, 'new_im_sigma200_typemanip1.png')
