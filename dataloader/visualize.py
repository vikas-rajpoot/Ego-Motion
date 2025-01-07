# import matplotlib.pyplot as plt

# def visualize_images_and_matrices(tgt_thr_img, ref_thr_imgs, tgt_thr_img_clr, ref_thr_img_clr, tgt_rgb_img, ref_rgb_imgs, intrinsics_thr, intrinsics_rgb, extrinsics_thr2rgb):
#     # Visualize images
#     def show_image(img, title):
#         plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
#         plt.title(title)
#         plt.axis('off')
#         plt.show()

#     show_image(tgt_thr_img, 'Target Thermal Image')
    
#     for i, img in enumerate(ref_thr_imgs):
#         show_image(img, f'Reference Thermal Image {i+1}')
    
#     show_image(tgt_thr_img_clr, 'Target Thermal Image Color')
    
#     for i, img in enumerate(ref_thr_img_clr):
#         show_image(img, f'Reference Thermal Image Color {i+1}')
    
#     show_image(tgt_rgb_img, 'Target RGB Image')
    
#     for i, img in enumerate(ref_rgb_imgs):
#         show_image(img, f'Reference RGB Image {i+1}')
    
#     # Visualize matrices
#     def show_matrix(matrix, title):
#         plt.imshow(matrix, cmap='viridis')
#         plt.colorbar()
#         plt.title(title)
#         plt.show()

#     show_matrix(intrinsics_thr, 'Intrinsics Thermal')
#     show_matrix(intrinsics_rgb, 'Intrinsics RGB')
#     show_matrix(extrinsics_thr2rgb, 'Extrinsics Thermal to RGB')

# # Example usage with dummy data
# # visualize_images_and_matrices(tgt_thr_img, ref_thr_imgs, tgt_thr_img_clr, ref_thr_img_clr, tgt_rgb_img, ref_rgb_imgs, intrinsics_thr, intrinsics_rgb, extrinsics_thr2rgb)