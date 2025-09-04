from cryolike import configure_likelihood_files, configure_displacement, run_likelihood_optimal_pose

for i_template in range(2):

    file_mgr = configure_likelihood_files(
        folder_templates = "./output/templates/",
        folder_particles = "./output/particles/",
        folder_output = "./output/likelihood/",
        n_stacks = 1,
        i_template = i_template
    )

    displacer = configure_displacement(
        max_displacement_pixels = 8.0,
        n_displacements_x = 16,
        n_displacements_y = 16,
    )

    run_likelihood_optimal_pose(
        file_config = file_mgr,
        params_input = "./output/templates/parameters.npz",
        displacer = displacer,
        template_index = i_template,
        n_stacks = 1,
        skip_exist = False,
        n_templates_per_batch = 16,
        n_images_per_batch = 64,
        estimate_batch_size = False,
        return_likelihood_optimal_pose_fourier = True,
        return_likelihood_integrated_pose_fourier = True
    )
