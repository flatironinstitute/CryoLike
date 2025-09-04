from cryolike import convert_particle_stacks_from_paired_star_and_mrc_files

pixel_size = 1.346
dataset_name = "apoferritin"
particle_file_list = ["./data/particles/particles.mrcs"]
star_file = ["./data/particles/particle_data.star"]

convert_particle_stacks_from_paired_star_and_mrc_files(
    params_input= "./output/templates/parameters.npz",
    folder_output = "./output/particles/",
    particle_file_list = particle_file_list,
    star_file_list = star_file,
    pixel_size = pixel_size,
    defocus_angle_is_degree = True,
    phase_shift_is_degree = True,
    skip_exist = False,
    flag_plots = True
)
