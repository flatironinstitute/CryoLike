from cryolike.util import post_process_output

post_process_output.stitch_log_likelihood_matrices(
    n_templates = 2,
    n_image_stacks=1,
    phys=False,
    output_directory='output/',
    cc=True,
    opt=True,
    integrated=True)
