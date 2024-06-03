import argparse
import numpy as np
import SimpleITK as sitk
import os

from projects.dental_det_infer import dental_detection
from detection3d.utils.landmark_utils import is_voxel_coordinate_valid, is_world_coordinate_valid
from segmentation3d.core.seg_infer import segmentation, segmentation_internal, load_models, check_input
from segmentation3d.utils.dicom_helper import write_binary_dicom_series
from segmentation3d.utils.dicom_helper import read_dicom_series, write_dicom_series, dicom_tags_dict
from segmentation3d.utils.image_tools import crop_image


def dental_refine_facial_region(input_dicom_folder, seg_model_folder, gpu_id,
                                left_face_anchor, right_face_anchor):
    """
    This function aims to refine the segmentation results in the facial region.
    """
    is_dicom_folder, file_name_list, file_path_list = check_input(input_dicom_folder)

    if is_dicom_folder:
        image = read_dicom_series(input_dicom_folder)
        image = sitk.Cast(image, sitk.sitkFloat32)
    else:
        image = sitk.ReadImage(file_path_list[0])

    crop_physical_volume = [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [image.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx] for idx in range(3)]

    models = load_models(seg_model_folder, gpu_id)

    left_face_mask = None
    if is_world_coordinate_valid(left_face_anchor):
        left_voxel_coord = image.TransformPhysicalPointToIndex(left_face_anchor)
        if is_voxel_coordinate_valid(left_voxel_coord, image.GetSize()):
            cropped_left_image = crop_image(image, left_face_anchor, crop_size, crop_spacing, 'LINEAR')
            _, left_face_mask = segmentation_internal(cropped_left_image, models, gpu_id)

    right_face_mask = None
    if is_world_coordinate_valid(right_face_anchor):
        right_voxel_coord = image.TransformPhysicalPointToIndex(right_face_anchor)
        if is_voxel_coordinate_valid(right_voxel_coord, image.GetSize()):
            cropped_right_image = crop_image(image, right_face_anchor, crop_size, crop_spacing, 'LINEAR')
            _, right_face_mask = segmentation_internal(cropped_right_image, models, gpu_id)

    return left_face_mask, right_face_mask


def dental_refine_bone_region(input_dicom_folder, seg_model_folder, gpu_id, left_bone_anchor, right_bone_anchor):
    """
    This function aims to refine the segmentation results in the facial region.
    """
    is_dicom_folder, file_name_list, file_path_list = check_input(input_dicom_folder)

    if is_dicom_folder:
        image = read_dicom_series(input_dicom_folder)
        image = sitk.Cast(image, sitk.sitkFloat32)
    else:
        image = sitk.ReadImage(file_path_list[0])

    crop_physical_volume = [38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [image.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx] for idx in range(3)]

    models = load_models(seg_model_folder, gpu_id)

    left_bone_mask = None
    if is_world_coordinate_valid(left_bone_anchor):
        left_voxel_coord = np.array(image.TransformPhysicalPointToIndex(left_bone_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, image.GetSize()):
            image_spacing = image.GetSpacing()
            voxel_offset = np.array([15 * 0.3 / image_spacing[0], 65 * 0.3 / image_spacing[1], -15 * 0.3 / image_spacing[2]])
            cropped_center = image.TransformContinuousIndexToPhysicalPoint(left_voxel_coord + voxel_offset)
            cropped_left_image = crop_image(image, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, left_bone_mask = segmentation_internal(cropped_left_image, models, gpu_id)

    right_bone_mask = None
    if is_world_coordinate_valid(right_bone_anchor):
        right_voxel_coord = np.array(image.TransformPhysicalPointToIndex(right_bone_anchor))
        if is_voxel_coordinate_valid(right_voxel_coord, image.GetSize()):
            image_spacing = image.GetSpacing()
            voxel_offset = np.array([-15 * 0.3 / image_spacing[0], 65 * 0.3 / image_spacing[1], -15 * 0.3 / image_spacing[2]])
            cropped_center = image.TransformContinuousIndexToPhysicalPoint(right_voxel_coord + voxel_offset)
            cropped_right_image = crop_image(image, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, right_bone_mask = segmentation_internal(cropped_right_image, models, gpu_id)

    return left_bone_mask, right_bone_mask

def dental_refine_IC_region(input_dicom_folder, seg_model_folder, gpu_id, palate_anchor):

    """
    This function aims to refine the segmentation results in the palate  region.
    """
    is_dicom_folder, file_name_list, file_path_list = check_input(input_dicom_folder)

    if is_dicom_folder:
        image = read_dicom_series(input_dicom_folder)
        image = sitk.Cast(image, sitk.sitkFloat32)
    else:
        image = sitk.ReadImage(file_path_list[0])

    crop_physical_volume = [48, 48, 48] #[38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [image.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx] for idx in range(3)]

    models = load_models(seg_model_folder, gpu_id)

    palate_mask = None
    if is_world_coordinate_valid(palate_anchor):
        left_voxel_coord = np.array(image.TransformPhysicalPointToIndex(palate_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, image.GetSize()):
            image_spacing = image.GetSpacing()
            voxel_offset = np.array([45 * 0.3 / image_spacing[0], -40 * 0.3 / image_spacing[1], 0])
            cropped_center = image.TransformContinuousIndexToPhysicalPoint(left_voxel_coord + voxel_offset)
            cropped_palate_image = crop_image(image, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, palate_mask = segmentation_internal(cropped_palate_image, models, gpu_id)


    return palate_mask

def dental_refine_whole_region(input_dicom_folder, seg_model_folder, gpu_id, A_anchor):

    """
    This function aims to refine the segmentation results in the palate  region.
    """
    is_dicom_folder, file_name_list, file_path_list = check_input(input_dicom_folder)

    if is_dicom_folder:
        image = read_dicom_series(input_dicom_folder)
        image = sitk.Cast(image, sitk.sitkFloat32)
    else:
        image = sitk.ReadImage(file_path_list[0])

    crop_physical_volume = [134.4, 72, 60] #[38.4, 38.4, 38.4]  # unit mm
    crop_spacing = [image.GetSpacing()[idx] for idx in range(3)]
    crop_size = [crop_physical_volume[idx] // crop_spacing[idx] for idx in range(3)]

    models = load_models(seg_model_folder, gpu_id)

    A_mask = None
    if is_world_coordinate_valid(A_anchor):
        left_voxel_coord = np.array(image.TransformPhysicalPointToIndex(A_anchor))
        if is_voxel_coordinate_valid(left_voxel_coord, image.GetSize()):
            image_spacing = image.GetSpacing()
            voxel_offset = np.array([0 * 0.3 / image_spacing[0], 100 * 0.3 / image_spacing[1], 30 * 0.3 / image_spacing[1]])
            cropped_center = image.TransformContinuousIndexToPhysicalPoint(left_voxel_coord + voxel_offset)
            cropped_palate_image = crop_image(image, cropped_center, crop_size, crop_spacing, 'LINEAR')
            _, A_mask = segmentation_internal(cropped_palate_image, models, gpu_id)


    return A_mask


def dental_det_seg(input_dicom_folder, det_model_folder, seg_model_folder, save_dicom_folder, gpu_id):

    """
    This interface is only used for the integration of auto-segmentation function into AA software.
    :param input_dicom_folder:        The input dicom folder
    :param det_model_folder:          The folder contains trained detection models
    :param seg_model_folder:          The folder contains trained segmentation models
    :param save_dicom_folder:         The folder to save binary masks of mandible and midfacec in dicom format
    :param gpu_id:                    Which gpu to use, by default, 0
    :return: None
    """
    assert os.path.isdir(det_model_folder) and os.path.isdir(seg_model_folder)

    det_model_folder = os.path.join(det_model_folder, 'model_0514_2020')
    seg_global_model_folder = os.path.join(seg_model_folder, 'model_0429_2020')
    seg_local_model1_folder = os.path.join(seg_model_folder, 'model_1115_2020')  # facial local refinement
    seg_local_model2_folder = os.path.join(seg_model_folder, 'model_0208_2021')

    seg_local_model3_folder = os.path.join(seg_model_folder, 'model_IC')
    seg_local_model4_folder = os.path.join(seg_model_folder, 'model_whole')

    #  ###### debug ############
    seg_local_model5_folder = os.path.join(seg_model_folder, 'model_whole_2')
    seg_local_model6_folder = os.path.join(seg_model_folder, 'model_whole_4')
    seg_local_model_COR_folder = os.path.join(seg_model_folder, 'model_COR')
    #  #################

    seg_softtissue_model_folder = os.path.join(seg_model_folder, 'model_0609_2020')
    seg_teeth_model_folder = os.path.join(seg_model_folder, 'model_0803_2020_dental')

    assert os.path.isdir(det_model_folder) and os.path.isdir(seg_global_model_folder) and \
           os.path.isdir(seg_local_model1_folder) and os.path.isdir(seg_local_model2_folder)

    # landmark detection and segmentation
    is_dicom_folder, file_name_list, file_path_list = check_input(input_dicom_folder)
    if is_dicom_folder:
        file_path_list = [input_dicom_folder]
        file_name_list = [os.path.split(input_dicom_folder)[-1]]

    for idx, file_path in enumerate(file_path_list):

        landmark_dataframes = dental_detection(file_path, det_model_folder, 0, gpu_id, save_dicom_folder)

        left_face_anchor = landmark_dataframes[0][landmark_dataframes[0]['name'] == 'ION-L']
        left_face_anchor= [left_face_anchor['x'].values[0], left_face_anchor['y'].values[0],
                           left_face_anchor['z'].values[0]]

        right_face_anchor = landmark_dataframes[0][landmark_dataframes[0]['name'] == 'ION-R']
        right_face_anchor= [right_face_anchor['x'].values[0], right_face_anchor['y'].values[0],
                            right_face_anchor['z'].values[0]]

        left_COR_anchor = landmark_dataframes[0][landmark_dataframes[0]['name'] == 'COR-L']
        left_COR_anchor= [left_COR_anchor['x'].values[0], left_COR_anchor['y'].values[0],
                           left_COR_anchor['z'].values[0]]

        right_COR_anchor = landmark_dataframes[0][landmark_dataframes[0]['name'] == 'COR-R']
        right_COR_anchor= [right_COR_anchor['x'].values[0], right_COR_anchor['y'].values[0],
                            right_COR_anchor['z'].values[0]]

        palate_anchor = landmark_dataframes[0][landmark_dataframes[0]['name'] == 'GPF-R']
        palate_anchor = [palate_anchor['x'].values[0], palate_anchor['y'].values[0],
                             palate_anchor['z'].values[0]]

        A_anchor = landmark_dataframes[0][landmark_dataframes[0]['name'] == 'A']
        A_anchor = [A_anchor['x'].values[0], A_anchor['y'].values[0], A_anchor['z'].values[0]]

        # segment facial regions
        left_face_mask, right_face_mask = dental_refine_facial_region(file_path, seg_local_model1_folder,
                                                                      gpu_id, left_face_anchor, right_face_anchor)
        sitk.WriteImage(left_face_mask, os.path.join(save_dicom_folder, 'left_face.nii.gz'), True)

        # segment bony structures behind facial regions
        left_bone_mask, right_bone_mask = dental_refine_bone_region(file_path, seg_local_model2_folder,
                                                                    gpu_id, left_face_anchor, right_face_anchor)
        sitk.WriteImage(left_bone_mask, os.path.join(save_dicom_folder, 'left_bone.nii.gz'), True)

        # segment condyle regions
        left_COR_mask, right_COR_mask = dental_refine_facial_region(file_path, seg_local_model_COR_folder,
                                                                    gpu_id, left_COR_anchor, right_COR_anchor)

        # segment palate structures
        # palate_mask = dental_refine_IC_region(file_path, seg_local_model3_folder,
        #                                           gpu_id, palate_anchor)
        # sitk.WriteImage(palate_mask, os.path.join(save_dicom_folder, 'palate.nii.gz'), True)

        # segment the whole patch
        A_mask = dental_refine_whole_region(file_path, seg_local_model4_folder,
                                              gpu_id, A_anchor)
        sitk.WriteImage(A_mask, os.path.join(save_dicom_folder, 'A_only.nii.gz'), True)
        # A_mask_2= dental_refine_whole_region(file_path, seg_local_model5_folder,
        #                                     gpu_id, A_anchor)
        # A_mask_4 = dental_refine_whole_region(file_path, seg_local_model6_folder,
        #                                     gpu_id, A_anchor)

        # segment the whole volume
        mask = segmentation(file_path, seg_global_model_folder, '', '', gpu_id, True, False, False, False)[0]
        #
        mask_name = file_name_list[idx]
        sitk.WriteImage(mask, os.path.join(save_dicom_folder, 'seg_pre.nii.gz'), True) # .format(mask_name)


        # Debug Hdeng
        # segment soft tissue and teeth
        # mask_face = segmentation(file_path, seg_softtissue_model_folder, '', '', gpu_id, True, False, False, False)[0]
        # sitk.WriteImage(mask_face, os.path.join(save_dicom_folder, 'test.nii.gz'), True)
        # mask_teeth = segmentation(file_path, seg_teeth_model_folder, '', '', gpu_id, True, False, False, False)[0]
        # sitk.WriteImage(mask_teeth, os.path.join(save_dicom_folder, 'test_teeth.nii.gz'), True)
        # End Edit

        # # merge the local and global results
#         mask_backup=mask
#         if A_mask is not None:
#             A_mask = A_mask[:, :, :-2]
#             A_start_voxel = mask.TransformPhysicalPointToIndex(A_mask.GetOrigin())
#             # mask_A_fill = sitk.Paste(mask, A_mask, A_mask.GetSize(), [0, 0, 0], A_start_voxel)
#             # sitk.WriteImage(mask_A_fill, os.path.join(save_dicom_folder, 'seg_A_fill.nii.gz'), True) #.format(mask_name)
#
#
#             # ### debug only ### test mask combine using xor
#
#             if is_world_coordinate_valid(A_anchor):
#                 left_voxel_coord = np.array(mask.TransformPhysicalPointToIndex(A_anchor))
#                 if is_voxel_coordinate_valid(left_voxel_coord, mask.GetSize()):
#                     image_spacing = mask.GetSpacing()
#                     voxel_offset = np.array(
#                         [0 * 0.3 / image_spacing[0], 100 * 0.3 / image_spacing[1], 30 * 0.3 / image_spacing[1]])
#                     cropped_center = mask.TransformContinuousIndexToPhysicalPoint(A_start_voxel)
#                     crop_size = A_mask.GetSize()
#                     cropping_origin = A_mask.GetOrigin()
#                     # crop_spacing = A_mask
#
#
#                     cropping_direction = mask.GetDirection()
#
#                     # if interp_method == 'LINEAR':
#                     #     interp_method = sitk.sitkLinear
#                     # elif interp_method == 'NN':
#                     #     interp_method = sitk.sitkNearestNeighbor
#                     # else:
#                     #     raise ValueError('Unsupported interpolation type.')
#
#                     transform = sitk.Transform(3, sitk.sitkIdentity)
#                     cropped_A_mask = sitk.Resample(mask, crop_size, transform, sitk.sitkNearestNeighbor, cropping_origin, image_spacing,
#                                                    cropping_direction)
#                     # cropped_A_mask = crop_image(mask, cropped_center, crop_size, image_spacing, 'LINEAR')
#
#
#             mask_A_xor = sitk.Or(cropped_A_mask, A_mask) #, A_mask.GetSize(), [0, 0, 0], A_start_voxel)
#             arr1 = sitk.GetArrayFromImage(cropped_A_mask)
#             arr2 = sitk.GetArrayFromImage(A_mask)
#             # arr = np.zeros_like(arr1)
#             # arr[arr1]
#             arr = np.maximum(arr1, arr2).astype(np.int8)
# #            arr = np.logical_or(arr1, arr2).astype(np.int8)
#             mask_A_xor = sitk.GetImageFromArray(arr)
#             mask_A_xor.CopyInformation(A_mask)
#
#             # sitk.WriteImage(mask_A_xor, os.path.join(save_dicom_folder, '{}_A_XOR_pre.nii.gz'.format(mask_name)), True)
#             mask = sitk.Paste(mask, mask_A_xor, mask_A_xor.GetSize(), [0, 0, 0], A_start_voxel)
#             sitk.WriteImage(mask, os.path.join(save_dicom_folder, 'seg_A_XOR.nii.gz'), True) #.format(mask_name)
#
# #
#
#             # ### end debug#######


            # A_start_voxel = mask_backup.TransformPhysicalPointToIndex(A_mask_2.GetOrigin())
            # mask_A_fill_2 = sitk.Paste(mask_backup, A_mask_2, A_mask_2.GetSize(), [0, 0, 0], A_start_voxel)
            # sitk.WriteImage(mask_A_fill_2, os.path.join(save_dicom_folder, '{}_A2_fill.nii.gz'.format(mask_name)), True)
            #
            # A_start_voxel = mask_backup.TransformPhysicalPointToIndex(A_mask_4.GetOrigin())
            # mask_A_fill_4 = sitk.Paste(mask_backup, A_mask_4, A_mask_4.GetSize(), [0, 0, 0], A_start_voxel)
            # sitk.WriteImage(mask_A_fill_4, os.path.join(save_dicom_folder, '{}_A4_fill.nii.gz'.format(mask_name)), True)

            # sitk.WriteImage(A_mask, os.path.join(save_dicom_folder, 'A_fill.nii.gz'), True)

        #
        if left_face_mask is not None:
            left_face_start_voxel = mask.TransformPhysicalPointToIndex(left_face_mask.GetOrigin())
            mask = sitk.Paste(mask, left_face_mask, left_face_mask.GetSize(), [0, 0, 0], left_face_start_voxel)

        if right_face_mask is not None:
            right_face_start_voxel = mask.TransformPhysicalPointToIndex(right_face_mask.GetOrigin())
            mask = sitk.Paste(mask, right_face_mask, right_face_mask.GetSize(), [0, 0, 0], right_face_start_voxel)


        # if left_COR_mask is not None:
        #     left_COR_mask = left_COR_mask[32:-32, 32:-32, 32:-32]
        #     left_COR_start_voxel = mask.TransformPhysicalPointToIndex(left_COR_mask.GetOrigin())
        #     mask = sitk.Paste(mask, left_COR_mask, left_COR_mask.GetSize(), [0, 0, 0], left_COR_start_voxel)
            #
            # if is_world_coordinate_valid(left_COR_anchor):
            #     left_voxel_coord = np.array(mask.TransformPhysicalPointToIndex(left_COR_anchor))
            #     if is_voxel_coordinate_valid(left_voxel_coord, mask.GetSize()):
            #         image_spacing = mask.GetSpacing()
            #         cropped_center = mask.TransformContinuousIndexToPhysicalPoint(left_COR_start_voxel)
            #         crop_size = left_COR_mask.GetSize()
            #         cropping_origin = left_COR_mask.GetOrigin()
            #         # crop_spacing = A_mask
            #
            #         cropping_direction = mask.GetDirection()
            #         transform = sitk.Transform(3, sitk.sitkIdentity)
            #         cropped_left_COR_mask = sitk.Resample(mask, crop_size, transform, sitk.sitkNearestNeighbor,
            #                                        cropping_origin, image_spacing,
            #                                        cropping_direction)
            #
            # arr1 = sitk.GetArrayFromImage(cropped_left_COR_mask)
            # arr2 = sitk.GetArrayFromImage(left_COR_mask)
            # arr = np.maximum(arr1, arr2).astype(np.int8)
            # #            arr = np.logical_or(arr1, arr2).astype(np.int8)
            # mask_left_COR_xor = sitk.GetImageFromArray(arr)
            # mask_left_COR_xor.CopyInformation(left_COR_mask)
            #
            # # sitk.WriteImage(mask_A_xor, os.path.join(save_dicom_folder, '{}_A_XOR_pre.nii.gz'.format(mask_name)), True)
            # mask = sitk.Paste(mask, mask_left_COR_xor, mask_left_COR_xor.GetSize(), [0, 0, 0], left_COR_start_voxel)


        # if right_COR_mask is not None:
        #     right_COR_mask = right_COR_mask[32:-32, 32:-32, 32:-32]
        #     right_COR_start_voxel = mask.TransformPhysicalPointToIndex(right_COR_mask.GetOrigin())
        #     mask = sitk.Paste(mask, right_COR_mask, right_COR_mask.GetSize(), [0, 0, 0], right_COR_start_voxel)
            #
            # if is_world_coordinate_valid(right_COR_anchor):
            #     right_voxel_coord = np.array(mask.TransformPhysicalPointToIndex(right_COR_anchor))
            #     if is_voxel_coordinate_valid(right_voxel_coord, mask.GetSize()):
            #         image_spacing = mask.GetSpacing()
            #         cropped_center = mask.TransformContinuousIndexToPhysicalPoint(right_COR_start_voxel)
            #         crop_size = right_COR_mask.GetSize()
            #         cropping_origin = right_COR_mask.GetOrigin()
            #         # crop_spacing = A_mask
            #
            #         cropping_direction = mask.GetDirection()
            #         transform = sitk.Transform(3, sitk.sitkIdentity)
            #         cropped_right_COR_mask = sitk.Resample(mask, crop_size, transform, sitk.sitkNearestNeighbor,
            #                                                cropping_origin, image_spacing,
            #                                                cropping_direction)
            #
            # arr1 = sitk.GetArrayFromImage(cropped_right_COR_mask)
            # arr2 = sitk.GetArrayFromImage(right_COR_mask)
            # arr = np.maximum(arr1, arr2).astype(np.int8)
            # mask_right_COR_xor = sitk.GetImageFromArray(arr)
            # mask_right_COR_xor.CopyInformation(right_COR_mask)
            #
            # mask = sitk.Paste(mask, mask_right_COR_xor, mask_right_COR_xor.GetSize(), [0, 0, 0], right_COR_start_voxel)

        #
        if left_bone_mask is not None:

            mask1 = mask
            mask2 = mask

            # left_bone_mask = left_bone_mask[:, 20:-10, :-5]
            left_bone_mask = left_COR_mask[32:-32, 32:-32, 32:-32]
            left_bone_start_voxel = mask.TransformPhysicalPointToIndex(left_bone_mask.GetOrigin())
            mask = sitk.Paste(mask, left_bone_mask, left_bone_mask.GetSize(), [0, 0, 0], left_bone_start_voxel)

            # left_bone_mask1 = left_bone_mask[:, 40:, :]
            # left_bone_start_voxel = mask1.TransformPhysicalPointToIndex(left_bone_mask1.GetOrigin())
            # mask1 = sitk.Paste(mask1, left_bone_mask1, left_bone_mask1.GetSize(), [0, 0, 0], left_bone_start_voxel)
            # sitk.WriteImage(mask1, os.path.join(save_dicom_folder, '{}_40end.nii.gz'.format(mask_name)), True)
            #
            # left_bone_mask1 = left_bone_mask[:, 30:-10, :-5]
            # left_bone_start_voxel = mask1.TransformPhysicalPointToIndex(left_bone_mask1.GetOrigin())
            # mask1 = sitk.Paste(mask1, left_bone_mask1, left_bone_mask1.GetSize(), [0, 0, 0], left_bone_start_voxel)
            # sitk.WriteImage(mask1, os.path.join(save_dicom_folder, '{}_3010.nii.gz'.format(mask_name)), True)
            #
            # left_bone_mask = left_bone_mask[:, 50:-32, :]
            # left_bone_start_voxel = mask2.TransformPhysicalPointToIndex(left_bone_mask.GetOrigin())
            # mask2 = sitk.Paste(mask2, left_bone_mask, left_bone_mask.GetSize(), [0, 0, 0], left_bone_start_voxel)
            # sitk.WriteImage(mask2, os.path.join(save_dicom_folder, '{}_50.nii.gz'.format(mask_name)), True)

        if right_bone_mask is not None:
            # right_bone_mask = right_bone_mask[:, 30:-10, :-5]
            right_bone_mask = right_bone_mask[32:-32, 32:-32, 32:-32]
            right_bone_start_voxel = mask.TransformPhysicalPointToIndex(right_bone_mask.GetOrigin())
            mask = sitk.Paste(mask, right_bone_mask, right_bone_mask.GetSize(), [0, 0, 0], right_bone_start_voxel)


        #
        # if palate_mask is not None:
        #     palate_mask = palate_mask[:, :, 32:-32]
        #     palate_start_voxel = mask.TransformPhysicalPointToIndex(palate_mask.GetOrigin())
        #     mask = sitk.Paste(mask, palate_mask, palate_mask.GetSize(), [0, 0, 0], palate_start_voxel)
        #
        # write_binary_dicom_series(mask, os.path.join(save_dicom_folder, '{}_midface'.format(mask_name)), 1, 100)
        # write_binary_dicom_series(mask, os.path.join(save_dicom_folder, '{}_mandible'.format(mask_name)), 2, 100)
        #
        # mask_teeth = segmentation(file_path, seg_teeth_model_folder, '', '', gpu_id, True, True, False, False)[0]
        # write_binary_dicom_series(mask_teeth, os.path.join(save_dicom_folder, '{}_upper'.format(mask_name)), 1, 100)
        # write_binary_dicom_series(mask_teeth, os.path.join(save_dicom_folder, '{}_lower'.format(mask_name)), 2, 100)

        # write_binary_dicom_series(mask, os.path.join(save_dicom_folder, '{}_st'.format(mask_name)), 3, 100)
        # write_binary_dicom_series(mask_face, os.path.join(save_dicom_folder, '{}_softtissue'.format(mask_name)), 1, 100)
        # write_binary_dicom_series(mask_teeth, os.path.join(save_dicom_folder, '{}_upper'.format(mask_name)), 1, 100)
        # write_binary_dicom_series(mask_teeth, os.path.join(save_dicom_folder, '{}_lower'.format(mask_name)), 2, 100)

        # # debug only
        sitk.WriteImage(mask, os.path.join(save_dicom_folder, 'seg.nii.gz'), True) #.format(mask_name)
        # sitk.WriteImage(left_face_mask, os.path.join(save_dicom_folder, '{}_left.nii.gz'.format(mask_name)), True)
        # sitk.WriteImage(right_face_mask, os.path.join(save_dicom_folder, '{}_right.nii.gz'.format(mask_name)), True)


def main():

    long_description = 'Inference interface for dental segmentation and landmark detection.'

    default_input = 'F:\Share\FL_DICOM\FL038' # 'F:\Share/0719\MarkE' # 'F:\Share\FL_eval\FL002_dicom' #'F:\Share/0727\StevensW' # 'F:\Share\AutoSeg_test\Origin_DICOM\FL002'#  'F:\Share/0713/68'# 'F:\Share\Dr.Jacob\CVitoria'# 'F:\Share/0719\MarkE2015' # 'F:\Share/0727\ChanPM' #   'F:\Share/0719\ChungK' #  #'C:\Project\DentalEngine-main\Model-Zoo-master\Dental/test_data\case_67_org.mha'
    #default_input = '\shenlab\lab_stor6\projects\CT_Dental\dataset\segmentation\data_v1\test.csv' # only used for batch testing
    default_det_model = 'C:\Project\DentalEngine-main\Model-Zoo-master\Dental\detection\landmark'
    default_seg_model = 'C:\Project\DentalEngine-main\Model-Zoo-master\Dental\segmentation'
    default_output = 'F:\Share\FL_result\FL038_RESULT' #'F:\Share/0719\MarkE_RESULT0803' #  'F:\Share/0727\ObergJ_result0803' #'F:\Share/0727\StevensW_result'  'F:\Share/0713/result/68_0731' #  # 'F:\Share\AutoSeg_test\FL001_new_model' #'F:\Share\Dr.Jacob\CVitoria_result' # 'F:\Share/0719\MarkE_test_0726' #  'F:\Share/0727\ChanPM_result' #'F:\Share/0713/result/68_COR' #    'F:\Share/0719\ChungK_test' #   # 'F:\Share/0713/result/68_new'# 'F:\Share\AutoSeg_test/test_org' #'C:\Project\DentalEngine-main\Model-Zoo-master\Dental/test_data'
    #default_output = '\shenlab\lab_stor6\qinliu\projects\CT_Dental\results\benchmark_0208_2021 # only used for batch testing
    default_gpu_id = 1

    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-i', '--input', default=default_input, help='input dicom folder or volumetric data')
    parser.add_argument('--det_model', default=default_det_model, help='detection model folder')
    parser.add_argument('--seg_model', default=default_seg_model, help='segmentation model folder')
    parser.add_argument('-o', '--output', default=default_output, help='output dicom folder to save binary masks')
    parser.add_argument('-g', '--gpu_id', default=default_gpu_id, type=int,
                        help='the gpu id to run model, set to -1 if using cpu only.')

    args = parser.parse_args()


    case_list = os.listdir(default_input)
    for case in case_list:
        case_path = os.path.join(default_input,case)
        if not os.path.isdir(case_path):
            case_path = default_input
            save_path = default_output
        else:
            save_path = os.path.join(default_output, case)

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        if save_path == default_output:
            dental_det_seg(case_path, args.det_model, args.seg_model, save_path, args.gpu_id)
            break
        else:
            dental_det_seg(case_path, args.det_model, args.seg_model, save_path, args.gpu_id)


if __name__ == '__main__':
    main()
