from sklearn.datasets import fetch_lfw_people


def load_lfw_dataset(min_faces, resize_factor):
    """
    Load the LFW dataset and filter individuals with at least `min_faces_per_person` images and resize by `resize_factor`.
    """
    
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize_factor, color=True)

    return lfw_people.images, lfw_people.target, lfw_people.target_names
    