import matplotlib.pyplot as plt

import numpy as np
import gzip
from io import StringIO
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn import metrics

def parse_header_of_csv(csv_str):
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')
    assert columns[0] == 'timestamp'
    assert columns[-1] == 'label_source'

    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass;

    feature_names = columns[1:first_label_ind]
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        assert label.startswith('label:')
        label_names[li] = label.replace('label:', '')
        pass

    return (feature_names, label_names)


def parse_body_of_csv(csv_str, n_features):
    full_table = np.loadtxt(StringIO(csv_str), delimiter=',', skiprows=1)
    timestamps = full_table[:, 0].astype(int)
    X = full_table[:, 1:(n_features + 1)]
    trinary_labels_mat = full_table[:, (n_features + 1):-1]
    M = np.isnan(trinary_labels_mat)
    Y = np.where(M, 0, trinary_labels_mat) > 0.

    return (X, Y, M, timestamps)


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
    if label == 'SITTING':
        return 'Sitting'
    if label == 'LYING_DOWN':
        return 'Lying down'
    if label == 'SLEEPING':
        return 'Sleeping'

    if label.endswith('_'):
        label = label[:-1] + ')'
        pass;

    label = label.replace('__', ' (').replace('_', ' ')
    label = label[0] + label[1:].lower()
    label = label.replace('i m', 'I\'m')
    return label


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi, feat) in enumerate(feature_names):
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
    user_data_file = '/data/users/teewz1076/har/dataset/%s.features_labels.csv.gz' % uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rt') as fid:
        csv_str = fid.read().strip()
        pass

    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return (X, Y, M, timestamps, feature_names, label_names)


def validate_column_names_are_consistent(old_column_names, new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.")

    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci, old_column_names[ci], new_column_names[ci]))
        pass
    return


def read_multiple_users_data(uuids):
    feature_names = None
    feat_sensor_names = None
    label_names = None
    X_parts = []
    Y_parts = []
    M_parts = []
    timestamps_parts = []
    uuid_inds_parts = []
    for (ui, uuid) in enumerate(uuids):
        (X_i, Y_i, M_i, timestamps_i, feature_names_i, label_names_i) = read_user_data(uuid)
        if feature_names is None:
            feature_names = feature_names_i
            pass
            pass
        else:
            validate_column_names_are_consistent(feature_names, feature_names_i)
            pass
        if label_names is None:
            label_names = label_names_i
            pass
        else:
            validate_column_names_are_consistent(label_names, label_names_i)
            pass
        X_parts.append(X_i)
        Y_parts.append(Y_i)
        M_parts.append(M_i)
        timestamps_parts.append(timestamps_i)
        uuid_inds_parts.append(ui * np.ones(len(timestamps_i)))
        pass

    # Combine all the users' data:
    X = np.concatenate(tuple(X_parts), axis=0)
    Y = np.concatenate(tuple(Y_parts), axis=0)
    M = np.concatenate(tuple(M_parts), axis=0)
    timestamps = np.concatenate(tuple(timestamps_parts), axis=0)
    uuid_inds = np.concatenate(tuple(uuid_inds_parts), axis=0)

    return (X, Y, M, uuid_inds, timestamps, feature_names, label_names)


def project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names), dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)
        use_feature = np.logical_or(use_feature, is_from_sensor)
        pass
    X = X[:, use_feature]
    return X


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train, axis=0)
    std_vec = np.nanstd(X_train, axis=0)
    return (mean_vec, std_vec)


def standardize_features(X, mean_vec, std_vec):
    X_centralized = X - mean_vec.reshape((1, -1))
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1, -1))
    X_standard = X_centralized / normalizers
    return X_standard


def train_model(X_train, Y_train, M_train, feat_sensor_names, label_names, sensors_to_use, target_label):
    X_train = project_features_to_selected_sensors(X_train, feat_sensor_names, sensors_to_use)
    print("== Projected the features to %d features from the sensors: %s" % (
        X_train.shape[1], ', '.join(sensors_to_use)))

    (mean_vec, std_vec) = estimate_standardization_params(X_train)
    X_train = standardize_features(X_train, mean_vec, std_vec)

    label_ind = label_names.index(target_label)
    y = Y_train[:, label_ind]
    missing_label = M_train[:, label_ind]
    existing_label = np.logical_not(missing_label)

    X_train = X_train[existing_label, :]
    y = y[existing_label];

    X_train[np.isnan(X_train)] = 0.

    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label), sum(y), sum(np.logical_not(y))))

    lr_model = RandomForestClassifier(n_estimators=1000, oob_score = True, random_state=1)
    lr_model.fit(X_train, y)

    # Assemble all the parts of the model:
    model = { \
        'sensors_to_use': sensors_to_use, \
        'target_label': target_label, \
        'mean_vec': mean_vec, \
        'std_vec': std_vec, \
        'lr_model': lr_model}

    return model


def test_model(X_test, Y_test, M_test, timestamps, feat_sensor_names, label_names, target_label_test, model):
    print(
        "********************************************RESULTS FOR " + target_label_test + "********************************************")
    X_test = project_features_to_selected_sensors(X_test, feat_sensor_names, model['sensors_to_use'])
    print("== Projected the features to %d features from the sensors: %s" % (
        X_test.shape[1], ', '.join(model['sensors_to_use'])))

    X_test = standardize_features(X_test, model['mean_vec'], model['std_vec'])

    label_ind = label_names.index(model['target_label'])
    y = Y_test[:, label_ind]
    missing_label = M_test[:, label_ind]
    existing_label = np.logical_not(missing_label)

    X_test = X_test[existing_label, :]
    y = y[existing_label]
    timestamps = timestamps[existing_label]

    X_test[np.isnan(X_test)] = 0.

    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y), get_label_pretty_name(target_label_test), sum(y), sum(np.logical_not(y))))

    y_pred = model['lr_model'].predict(X_test)

    accuracy = np.mean(y_pred == y)

    tp = np.sum(np.logical_and(y_pred, y))
    tn = np.sum(np.logical_and(np.logical_not(y_pred), np.logical_not(y)))
    fp = np.sum(np.logical_and(y_pred, np.logical_not(y)))
    fn = np.sum(np.logical_and(np.logical_not(y_pred), y))

    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2.
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    '''
    ns_probs = [0 for _ in range(len(Y_test))]
    lr_probs = y_pred[:, 1]

    ns_auc = roc_auc_score(Y_test, ns_probs)
    lr_auc = roc_auc_score(Y_test, lr_probs)
    '''
    print("-" * 10)
    print('Accuracy*:         %.2f' % accuracy)
    print('Sensitivity (TPR): %.2f' % sensitivity)
    print('Specificity (TNR): %.2f' % specificity)
    print('Balanced accuracy: %.2f' % balanced_accuracy)
    print('Precision**:       %.2f' % precision)
    print('Recall:       %.2f' % recall)
    '''
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    '''
    print("-" * 10)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name = 'ROC for ' + target_label_test)
    display.plot()
    plt.savefig('ROC_RF500NewMetT' + target_label_test + '.png')
    plt.clf()
    '''
    disp = RocCurveDisplay.from_estimator(model, X_test, Y_test)
    plt.savefig('ROC_RF100NewMetT' + target_label_test + '.png')
    plt.clf()
    '''
    '''
    ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)

    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    plt.title('%s\nROC curve for' % get_label_pretty_name(model['target_label']));
    plt.savefig('ROC_RF500NewMetT' + target_label_test + '.png')
    plt.clf()
    '''

    fig = plt.figure(figsize=(10, 4), facecolor='white')
    ax = plt.subplot(1, 1, 1)
    ax.plot(timestamps[y], 1.4 * np.ones(sum(y)), '|g', markersize=10, label='ground truth')
    ax.plot(timestamps[y_pred], np.ones(sum(y_pred)), '|b', markersize=10, label='prediction')

    seconds_in_day = (60 * 60 * 24)
    tick_seconds = range(timestamps[0], timestamps[-1], seconds_in_day)
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int)

    ax.set_ylim([0.5, 5])
    # ax.set_xticks(tick_seconds)
    ax.set_xticklabels(tick_labels)
    plt.xlabel('days of participation', fontsize=14)
    ax.legend(loc='best')
    plt.title('%s\nGround truth vs. predicted' % get_label_pretty_name(model['target_label']));
    plt.savefig('RFNewMetT' + target_label_test + '.png')
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

uuids = ['00EABED2-271D-49D8-B599-1D4A09240601', '098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
         '0A986513-7828-4D53-AA1F-E02D6DF9561B', '0BFC35E2-4817-4865-BFA7-764742302A2D',
         '0E6184E1-90C0-48EE-B25A-F1ECB7B9714E', '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
         '11B5EC4D-4133-4289-B475-4E737182A406', '136562B6-95B2-483D-88DC-065F28409FD2',
         '1538C99F-BA1E-4EFB-A949-6C7C47701B20', '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
         '24E40C4C-A349-4F9F-93AB-01D00FB994AF', '27E04243-B138-4F40-A164-F40B60165CF3',
         '2C32C23E-E30C-498A-8DD2-0EFB9150A02E', '33A85C34-CFE4-4732-9E73-0A7AC861B27A',
         '3600D531-0C55-44A7-AE95-A7A38519464E', '40E170A7-607B-4578-AF04-F021C3B0384A',
         '481F4DD2-7689-43B9-A2AA-C8772227162B', '4E98F91F-4654-42EF-B908-A3389443F2E7',
         '4FC32141-E888-4BFF-8804-12559A491D8C', '5119D0F8-FCA8-4184-A4EB-19421A40DE0D',
         '5152A2DF-FAF3-4BA8-9CA9-E66B32671A53', '59818CD2-24D7-4D32-B133-24C2FE3801E5',
         '59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2', '5EF64122-B513-46AE-BCF1-E62AAC285D2C',
         '61359772-D8D8-480D-B623-7C636EAD0C81', '61976C24-1C50-4355-9C49-AAE44A7D09F6',
         '665514DE-49DC-421F-8DCB-145D0B2609AD', '74B86067-5D4B-43CF-82CF-341B76BEA0F4',
         '78A91A4E-4A51-4065-BDA7-94755F0BB3BB', '797D145F-3858-4A7F-A7C2-A4EB721E133C',
         '7CE37510-56D0-4120-A1CF-0E23351428D2', '7D9BB102-A612-4E2A-8E22-3159752F55D8',
         '8023FE1A-D3B0-4E2C-A57A-9321B7FC755F', '806289BC-AD52-4CC1-806C-0CDB14D65EB6',
         '81536B0A-8DBF-4D8A-AC24-9543E2E4C8E0', '83CF687B-7CEC-434B-9FE8-00C3D5799BE6',
         '86A4F379-B305-473D-9D83-FC7D800180EF', '96A358A0-FFF2-4239-B93E-C7425B901B47',
         '9759096F-1119-4E19-A0AD-6F16989C7E1C', '99B204C0-DD5C-4BB7-83E8-A37281B8D769',
         '9DC38D04-E82E-4F29-AB52-B476535226F2', 'A5A30F76-581E-4757-97A2-957553A2C6AA',
         'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96', 'A7599A50-24AE-46A6-8EA6-2576F1011D81',
         'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A', 'B09E373F-8A54-44C8-895B-0039390B859F',
         'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C', 'B9724848-C7E2-45F4-9B3F-A1F38D864495',
         'BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC', 'BEF6C611-50DA-4971-A040-87FB979F3FC1']
tempuuid = ['00EABED2-271D-49D8-B599-1D4A09240601', '098A72A5-E3E5-4F54-A152-BBDA0DF7B694']
(X, Y, M, uuid_inds, timestamps, feature_names, label_names) = read_multiple_users_data(uuids)

feat_sensor_names = get_sensor_names_from_features(feature_names);

'''
#TESTING: Prints users data
print("The parts of the concatenated users' data (and their dimensions):")
print("Every example has its timestamp, indicating the minute when the example was recorded")
print("%d users has %d examples (~%d minutes of behavior)" % (len(uuids),len(timestamps),len(timestamps)))
print(timestamps.shape)

n_examples_per_label = np.sum(Y,axis=0)
labels_and_counts = zip(label_names,n_examples_per_label)
sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1])

print ("How many examples do these users have for each contex-label:")
print ("-"*20)
for (label,count) in sorted_labels_and_counts:
    print( "label %s - %d minutes" % (label,count))
    pass

print("Features:") 
for (fi,feature) in enumerate(feature_names):
    print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature))
    pass
'''

sensors_to_use = ['Acc', 'Gyro', 'WAcc', 'watch_heading', 'location']
target_label = 'FIX_walking'
model_walk = train_model(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_label)
target_label = 'FIX_running'
model_run = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)
target_label = 'OR_standing'
model_standing = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)
target_label = 'LYING_DOWN'
model_lying = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)
target_label = 'SITTING'
model_sitting = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)
target_label = 'SLEEPING'
model_sleeping = train_model(X,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label)

testUUIDs = ['FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF', 'F50235E0-DD67-4F2A-B00B-1F31ADA998B9',
             'ECECC2AB-D32F-4F90-B74C-E12A1C69BBE2', 'E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3',
             'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B', 'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC',
             'CDA3BBF7-6631-45E8-85BA-EEB416B32A3C', 'CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F',
             'CA820D43-E5E2-42EF-9798-BE56F776370B', 'C48CE857-A0DD-4DDB-BEA5-3A25449B2153']
temptestuuid = ['FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF']
(X_test, Y_test, M_test, uuid_inds_test, timestamps_test, feature_names_test,
 label_names_test) = read_multiple_users_data(testUUIDs)
feat_sensor_names_test = get_sensor_names_from_features(feature_names_test);
target_label_test = 'FIX_walking'
test_model(X_test, Y_test, M_test, timestamps_test, feat_sensor_names_test, label_names_test, target_label_test,
           model_walk)
target_label_test = 'FIX_running'
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,target_label_test,model_run)
target_labe_testl = 'OR_standing'
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,target_label_test,model_standing)
target_label_test = 'LYING_DOWN'
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,target_label_test,model_lying)
target_label_test = 'SITTING'
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,target_label_test,model_sitting)
target_label_test = 'SLEEPING'
test_model(X_test,Y_test,M_test,timestamps_test,feat_sensor_names_test,label_names_test,target_label_test,model_sleeping)

print(
    '* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
print(
    '** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')