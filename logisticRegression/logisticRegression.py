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
        (X_i, Y_i, M_i, timestamps_i, feature_names_i, label_names_i) = read_user_data(uuid)
        if feature_names is None:
            feature_names = feature_names_i;
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

def project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names), dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature, is_from_sensor);
        pass;
    X = X[:, use_feature];
    return X;


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train, axis=0);
    std_vec = np.nanstd(X_train, axis=0);
    return (mean_vec, std_vec);


def standardize_features(X, mean_vec, std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1, -1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1, -1));
    X_standard = X_centralized / normalizers;
    return X_standard;


def train_model(X_train, Y_train, M_train, feat_sensor_names, label_names, sensors_to_use, target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train, feat_sensor_names, sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (
    X_train.shape[1], ', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec, std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train, mean_vec, std_vec);

    # The single target label:
    label_ind = label_names.index(target_label);
    y = Y_train[:, label_ind];
    missing_label = M_train[:, label_ind];
    existing_label = np.logical_not(missing_label);

    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label, :];
    y = y[existing_label];

    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.;

    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y))));

    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.

    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced', max_iter = 1000);
    lr_model.fit(X_train, y);

    # Assemble all the parts of the model:
    model = { \
        'sensors_to_use': sensors_to_use, \
        'target_label': target_label, \
        'mean_vec': mean_vec, \
        'std_vec': std_vec, \
        'lr_model': lr_model};

    return model;

def test_model(X_test, Y_test, M_test, timestamps, feat_sensor_names, label_names, model):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test, feat_sensor_names, model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (
    X_test.shape[1], ', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test, model['mean_vec'], model['std_vec']);

    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:, label_ind];
    missing_label = M_test[:, label_ind];
    existing_label = np.logical_not(missing_label);

    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label, :];
    y = y[existing_label];
    timestamps = timestamps[existing_label];

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;

    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y))));

    # Preform the prediction:
    y_pred = model['lr_model'].predict(X_test);

    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);

    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred, y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y));

    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp + fn);
    specificity = float(tn) / (tn + fp);

    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;

    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp + fp);

    print("-" * 10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print("-" * 10);

    print(
        '* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
    print(
        '** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')

    fig = plt.figure(figsize=(10, 4), facecolor='white');
    ax = plt.subplot(1, 1, 1);
    ax.plot(timestamps[y], 1.4 * np.ones(sum(y)), '|g', markersize=10, label='ground truth');
    ax.plot(timestamps[y_pred], np.ones(sum(y_pred)), '|b', markersize=10, label='prediction');

    seconds_in_day = (60 * 60 * 24);
    tick_seconds = range(timestamps[0], timestamps[-1], seconds_in_day);
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int);

    ax.set_ylim([0.5, 5]);
    ax.set_xticks(tick_seconds);
    ax.set_xticklabels(tick_labels);
    plt.xlabel('days of participation', fontsize=14);
    ax.legend(loc='best');
    plt.title('%s\nGround truth vs. predicted' % get_label_pretty_name(model['target_label']));
    plt.savefig('Logistic Regression.png')
    plt.clf()

    return;


'''
TESTING: Prints single user data

uuid = '00EABED2-271D-49D8-B599-1D4A09240601'
(X, Y, M, timestamps, feature_names, label_names) = read_user_data(uuid)
print("The parts of the user's data (and their dimensions):")
print("Every example has its timestamp, indicating the minute when the example was recorded")
print("User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps)))
print(timestamps.shape)
'''

uuids = ['00EABED2-271D-49D8-B599-1D4A09240601','098A72A5-E3E5-4F54-A152-BBDA0DF7B694','0A986513-7828-4D53-AA1F-E02D6DF9561B','0BFC35E2-4817-4865-BFA7-764742302A2D',
         '0E6184E1-90C0-48EE-B25A-F1ECB7B9714E']
(X, Y, M, uuid_inds, timestamps, feature_names, label_names) = read_multiple_users_data(uuids)

feat_sensor_names = get_sensor_names_from_features(feature_names);


#TESTING: Prints users data
print("The parts of the concatenated users' data (and their dimensions):")
print("Every example has its timestamp, indicating the minute when the example was recorded")
print("%d users has %d examples (~%d minutes of behavior)" % (len(uuids),len(timestamps),len(timestamps)))
print(timestamps.shape)

n_examples_per_label = np.sum(Y,axis=0);
labels_and_counts = zip(label_names,n_examples_per_label);
sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1]);

print ("How many examples do these users have for each contex-label:");
print ("-"*20);
for (label,count) in sorted_labels_and_counts:
    print( "label %s - %d minutes" % (label,count));
    pass;
   
print("Features:") 
for (fi,feature) in enumerate(feature_names):
    print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature));
    pass;


sensors_to_use = ['Acc','Gyro','WAcc','watch_heading','location']
target_label = 'FIX_walking'
model_walk = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
target_label = 'FIX_running'
model_run = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
target_label = 'OR_indoors'
model_indoors = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
target_label = 'OR_exercise'
model_exercise = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);
target_label = 'OR_standing'
model_standing = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);


testUUIDs = ['FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF','F50235E0-DD67-4F2A-B00B-1F31ADA998B9','ECECC2AB-D32F-4F90-B74C-E12A1C69BBE2','E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3',
         'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B']
(X_test, Y_test, M_test, uuid_inds_test, timestamps_test, feature_names_test, label_names_test) = read_multiple_users_data(testUUIDs)
feat_sensor_names_test = get_sensor_names_from_features(feature_names_test);
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,model_walk);
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,model_run);
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,model_indoors);
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,model_exercise);
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,model_standing);
