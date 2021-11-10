import matplotlib.pyplot as plt

import numpy as np
import gzip
from io import StringIO
import sklearn.linear_model

def parse_header_of_csv(csv_str):
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')
    assert columns[0] == 'timestamp';
    assert columns[-1] == 'label_source';

    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass;

    feature_names = columns[1:first_label_ind]
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        assert label.startswith('label:');
        label_names[li] = label.replace('label:', '')
        pass;

    return (feature_names, label_names);


def parse_body_of_csv(csv_str, n_features):
    full_table = np.loadtxt(StringIO(csv_str), delimiter=',', skiprows=1)
    timestamps = full_table[:, 0].astype(int);
    X = full_table[:, 1:(n_features + 1)];
    trinary_labels_mat = full_table[:, (n_features + 1):-1]
    M = np.isnan(trinary_labels_mat);
    Y = np.where(M, 0, trinary_labels_mat) > 0.

    return (X, Y, M, timestamps);

def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking'
    if label == 'FIX_running':
        return 'Running'
    if label == 'LOC_main_workplace':
        return 'At main workplace'
    if label == 'OR_indoors':
        return 'Indoors'
    if label == 'OR_outside':
        return 'Outside'
    if label == 'LOC_home':
        return 'At home'
    if label == 'FIX_restaurant':
        return 'At a restaurant'
    if label == 'OR_exercise':
        return 'Exercise'
    if label == 'LOC_beach':
        return 'At the beach'
    if label == 'OR_standing':
        return 'Standing'
    if label == 'WATCHING_TV':
        return 'Watching TV'

    if label.endswith('_'):
        label = label[:-1] + ')'
        pass;

    label = label.replace('__', ' (').replace('_', ' ')
    label = label[0] + label[1:].lower()
    label = label.replace('i m', 'I\'m')
    return label

def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)

        pass;

    return feat_sensor_names;

def read_user_data(uuid):
    user_data_file = '/data/users/teewz1076/har/dataset/%s.features_labels.csv.gz' % uuid;

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rt') as fid:
        csv_str = fid.read().strip();
        pass;

    (feature_names, label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features);

    return (X, Y, M, timestamps, feature_names, label_names);


def validate_column_names_are_consistent(old_column_names, new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.")

    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci, old_column_names[ci], new_column_names[ci]))
        pass;
    return;

'''
def read_multiple_users_data(uuids):
    feature_names = None;
    feat_sensor_names = None;
    label_names = None;
    X_parts = [];
    Y_parts = [];
    M_parts = [];
    timestamps_parts = [];
    uuid_inds_parts = [];
    for (ui, uuid) in enumerate(uuids):
        (X_i, Y_i, M_i, timestamps_i, feature_names_i, feat_sensor_names_i, label_names_i)
        if feature_names is None:
            feature_names = feature_names_i;
            feat_sensor_names = feat_sensor_names_i;
            pass;
        else:
            validate_column_names_are_consistent(feature_names, feature_names_i)
            pass;
        if label_names is None:
            label_names = label_names_i;
            pass;
        else:
            validate_column_names_are_consistent(label_names, label_names_i)
            pass;
        X_parts.append(X_i);
        Y_parts.append(Y_i);
        M_parts.append(M_i);
        timestamps_parts.append(timestamps_i);
        uuid_inds_parts.append(ui * np.ones(len(timestamps_i)));
        pass;

    # Combine all the users' data:
    X = np.concatenate(tuple(X_parts), axis=0);
    Y = np.concatenate(tuple(Y_parts), axis=0);
    M = np.concatenate(tuple(M_parts), axis=0);
    timestamps = np.concatenate(tuple(timestamps_parts), axis=0);
    uuid_inds = np.concatenate(tuple(uuid_inds_parts), axis=0);

    return (X, Y, M, uuid_inds, timestamps, feature_names, label_names)
'''

uuid = '00EABED2-271D-49D8-B599-1D4A09240601'
(X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid)
uuid2 =' 098A72A5-E3E5-4F54-A152-BBDA0DF7B694'
(X, Y, M, timestamps, feature_names, label_names) = (X, Y, M, timestamps, feature_names, label_names) (read_user_data(uuid2))

'''
TESTING: Prints user data'''
print("The parts of the user's data (and their dimensions):")
print("Every example has its timestamp, indicating the minute when the example was recorded")
print("User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps)))
print(timestamps.shape)

'''
(X, Y, M, uuid_inds, timestamps, feature_names, label_names) = read_multiple_users_data(['00EABED2-271D-49D8-B599-1D4A09240601', '098A72A5-E3E5-4F54-A152-BBDA0DF7B694'])


TESTING: Prints user data
print("The parts of the user's data (and their dimensions):")
print("Every example has its timestamp, indicating the minute when the example was recorded")
print("User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps)))
print(timestamps.shape)
'''