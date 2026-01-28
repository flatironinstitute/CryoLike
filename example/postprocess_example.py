from cryolike.util import post_process_output

post_process_output.stitch_log_likelihood_matrices(
    phys=False,
    output_directory='output/',
    cc=True,
    opt=True,
    integrated=True)
