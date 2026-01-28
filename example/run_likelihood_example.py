from cryolike.run_likelihood import run_likelihood_batch
    
for i_template in range(2):
    for i_stack in range(2):
    run_likelihood_batch(
        params_input = "./output/templates/parameters.npz",
        folder_templates = "./output/templates/",
        folder_particles = "./output/particles/",
        folder_output = "./output/likelihood/",
        i_template = i_template,
        i_stack = i_stack,
        skip_exist = False,
        n_templates_per_batch = 16,
        n_images_per_batch = 128,
        search_batch_size = True,
        max_displacement_pixels = 8.0,
        n_displacements_x = 16,
        n_displacements_y = 16,
        return_likelihood_integrated_pose_fourier = True,
        return_likelihood_optimal_pose_physical = False,
        return_likelihood_optimal_pose_fourier = True,
        verbose = True
    )
