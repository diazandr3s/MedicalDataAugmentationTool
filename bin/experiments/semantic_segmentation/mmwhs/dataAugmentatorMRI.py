import numpy as np
import SimpleITK as sitk
import os
import math
from scipy import ndimage

base_dir = '/usr/not-backed-up2/scsad/DL/MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/mmwhs/mmwhs_dataset/mr_mha'

output_dir = '/usr/not-backed-up2/scsad/DL/MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/mmwhs/mmwhs_dataset/mr_mha'

def get_list_of_files(base_dir):
    list_of_images = []
    list_of_labels = []
    patients = os.listdir(base_dir)
    for p in patients:
        if 'image' in p and p.startswith("mr_train_"):
            list_of_images.append(os.path.join(base_dir, p))
        elif 'label' in p and p.startswith("mr_train_"):
            list_of_labels.append(os.path.join(base_dir, p))
    print("Found %d patients" % len(list_of_images))
    return list_of_images, list_of_labels


def load_img(dir_img):
    # load SimpleITK Image
    img_sitk = sitk.ReadImage(dir_img)

    # get pixel arrays from SimpleITK images
    img_npy = sitk.GetArrayFromImage(img_sitk)

    # get some metadata
    spacing = img_sitk.GetSpacing()
    # the spacing returned by SimpleITK is in inverse order relative to the numpy array we receive.
    spacing = np.array(spacing)[::-1]
    direction = img_sitk.GetDirection()
    origin = img_sitk.GetOrigin()

    original_shape = img_npy.shape

    metadata = {
    'spacing': spacing,
    'direction': direction,
    'origin': origin,
    'original_shape': original_shape
    }


    return img_sitk, img_npy, metadata


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()

    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                          int(np.ceil(height/2)),
                                          int(np.ceil(depth/2))))


def rotate_img(img_sitk, transform, is_label, theta_x, theta_y, theta_z):
    # Angles for each axis
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)

    new_transform = sitk.Euler3DTransform(get_center(img_sitk), theta_x, theta_y, theta_z, (0, 0, 0))
    image_center = get_center(img_sitk)
    new_transform.SetCenter(image_center)
    new_transform.SetRotation(theta_x, theta_y, theta_z)

    # Resample
    reference_image = img_sitk

    if is_label:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkBSpline

    default_value = 0
    resampled = sitk.Resample(img_sitk, reference_image, new_transform,
                         interpolator, default_value)
    npy_img = sitk.GetArrayFromImage(resampled)

    return npy_img, resampled

def get_center_rai(label):

    center = ndimage.measurements.center_of_mass(label)

    return (round(center[2]), round(center[1]), round(center[0]))

def save_as_nii(img_npy, img_name, metadata):
    sitk_image = sitk.GetImageFromArray(img_npy)
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    # remember to revert spacing back to sitk order again
    sitk_image.SetSpacing(tuple(metadata['spacing'][[1, 2, 0]]))
    sitk.WriteImage(sitk_image, img_name)



list_of_images, list_of_labels = get_list_of_files(base_dir)

new_centers = []
name_aug_img_data = []

for i in range(0,len(list_of_images)):

    for deg in range(15, 40, 10):

        dir_img = list_of_images[i]
        dir_label = list_of_labels[i]

        name_image =  dir_img.split('/')[12:][0][0:-4]
        name_label =  dir_label.split('/')[12:][0][0:-4]

        print(name_image)
        print(name_label)

        # Loading image and label
        img_sitk, img_npy, metadata_img = load_img(dir_img)
        label_sitk, label_npy, metadata_label = load_img(dir_label)

        theta_x, theta_y, theta_z = deg, 0, deg

        affine_ro = sitk.AffineTransform(3)
        npy_rotated, rotated_image = rotate_img(img_sitk, affine_ro, False, theta_x, theta_y, theta_z)
        print(npy_rotated.shape)

        # Rotate label
        npy_rotated_label, rotated_label = rotate_img(label_sitk, affine_ro, True, theta_x, theta_y, theta_z)

        # Downsampling image.
        max_slices = math.floor(npy_rotated.shape[1] / 8)
        downsampled_img = np.array([npy_rotated[:,l*5,:] for l in range(0, max_slices)],dtype=np.float32)
        name_aug_img_data.append('augmented_' + str(deg) + name_image[:-6])
        # Changing the spacing
        metadata_img['spacing'] = np.array([8, 0.9, 0.9])
        save_as_nii(downsampled_img, output_dir + '/augmented_' + str(deg) + name_image + '.mha', metadata_img)

        # Downsampling label
        downsampled_label = np.array([npy_rotated_label[:,l*5,:] for l in range(0, max_slices)],dtype=np.int16)
        # Changing the spacing
        metadata_label['spacing'] = np.array([8, 0.9, 0.9])
        save_as_nii(downsampled_label, output_dir + '/augmented_' + str(deg) + name_label + '.mha', metadata_label)

        # Getting new segmentation center
        new_centers.append(get_center_rai(downsampled_label))



# Saving seg center rai
with open('mr_seg_center_rai_China.txt', 'w') as f:
    for i in range(len(new_centers)):
        f.write("%s,"%name_aug_img_data[i])
        f.write("%s,%s,%s\n"%new_centers[i])
