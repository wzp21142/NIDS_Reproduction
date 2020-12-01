import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Mark the nominal columns and consolidate the data (extract all the nominal columns)
# 标出名词性列，整合数据（把名词性的列都提取出来）
def combine_dataset(files, col_names, processed=False):
    dtypes = {}
    if processed == False:
        for col_name in col_names:
            nominal_names = set(['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state',
                                 'service', 'ct_ftp', 'label_10'])  # Nominal column
            if col_name in nominal_names:
                dtypes[col_name] = str
            else:
                dtypes[col_name] = np.float32
    else:
        for col_name in col_names:
            dtypes[col_name] = np.float32

    records = []
    for file in files:
        data = pd.read_csv(file, header=None, names=col_names, dtype=dtypes)
        records.append(data)

    records_all = pd.concat(records)  # 当没有索引时、concat不管列名，直接加到一起
    # When there is no index, concat adds them together regardless of the column names,

    return records_all


## Make new col names for categorical features after one-hot encoding
# 为one-hot编码之后的列起个新列名
def get_nominal_names(dataset, cols_nominal):
    data_nominal = dataset[cols_nominal]

    new_col_names = []
    for col_name in cols_nominal:
        name_unique = sorted(dataset[col_name].unique())  # 名词性列的不同的值。Different values for noun columns
        new_col_name = [col_name + '_' + x for x in name_unique]
        new_col_names.extend(new_col_name)

    return new_col_names


# Remove the unimportant feature, one-hot encoding, and convert the attack class to numeric
# 删除不重要的特征，one-hot编码，将攻击类别转换为数值型
def select_feature_and_encoding(dataset, cols_to_drop, cols_nominal, cols_nominal_all):
    # Drop the features has no meaning such as src ip. 删除不重要的特征
    for cols in cols_to_drop:
        dataset.drop(cols, axis=1, inplace=True)

    # Save the label and then drop it from dataset 保留标签然后将它从数据集中删除（提取出标签列）
    label_10 = dataset['label_10']
    label_2 = dataset['label_2']
    dataset.drop('label_2', axis=1, inplace=True)
    dataset.drop('label_10', axis=1, inplace=True)

    # replace the label with specific code  将标签数值化
    replace_dict = {np.nan: 0, 'Analysis': 1, 'Backdoors': 2, 'Backdoor': 2, 'DoS': 3,
                    'Exploits': 4, ' Fuzzers': 5, ' Fuzzers ': 5, 'Generic': 6,
                    'Reconnaissance': 7, ' Shellcode ': 8, 'Shellcode': 8,
                    'Worms': 9, ' Reconnaissance ': 7, }
    new_label_10 = label_10.replace(replace_dict)
    new_label_10.to_frame()
    label_2.to_frame
    del label_10

    # replace the lost values  用0替换缺失值
    replace_dict = {np.nan: 0, ' ': 0}
    for cols in ['ct_ftp', 'ct_flw', 'is_ftp']:
        dataset[cols] = dataset[cols].replace(replace_dict)

    # 'is_ftp' column is wrong, correct it(I found that the value of it is
    # all the same with ct_ftp_cmd, so if the value is not 0, is_ftp should
    # be 1)
    for x in dataset['is_ftp']:
        if x != 0:
            x = 1

    # select and process the categorical features 选择并处理分类特征
    data_nominal = dataset[cols_nominal]  # cols_nominal是名词性列的列名，提取出名词性列的数据
    data_temp_1 = data_nominal.apply(LabelEncoder().fit_transform)  # 将名词性列进行编号
    del data_nominal

    new_col_names = []
    for col_name in cols_nominal:
        name_unique = sorted(dataset[col_name].unique())
        new_col_name = [col_name + '_' + x for x in name_unique]

        new_col_names.extend(new_col_name)
        dataset.drop(col_name, axis=1, inplace=True)

    # one-hot
    enc = OneHotEncoder()
    data_temp_2 = enc.fit_transform(data_temp_1)
    del data_temp_1

    data_encoded = pd.DataFrame(data_temp_2.toarray(), columns=new_col_names)
    del data_temp_2

    # complement the nominal columns 补充名词性列
    diff = set(cols_nominal_all) - set(new_col_names)

    if diff:
        for cols in diff:
            data_encoded[cols] = 0.
        data_encoded = data_encoded[cols_nominal_all]

    dataset = dataset.join(data_encoded)
    del data_encoded

    dataset = dataset.join(new_label_10)
    dataset = dataset.join(label_2)

    return dataset  # Complete data set (including data and labels)
    # 完整的数据集（包括数据和标签）


# Split the training set and test set and save the file as a CSV file
# 分裂训练集和测试集,并将文件保存成CSV文件
def split_dataset(dataset, file_train, file_test):
    cols = dataset.columns
    # trainset, testset = train_test_split(dataset, test_size = 0.2)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=40, stratify=dataset['label_10'])
    train = pd.DataFrame(trainset, columns=cols)
    test = pd.DataFrame(testset, columns=cols)

    train.to_csv(file_train)
    test.to_csv(file_test)


# Standardize, and save the file in CSV and tf formats
# 标准化，并将文件保存成csv格式和tf格式
def scaling(files_train, files_test, col_names_scaling, scaling_type):
    if scaling_type == 'min_max':
        scaler = MinMaxScaler()
        file_folder = 'min_max/'
    else:
        scaler = StandardScaler()
        file_folder = 'normalized/'

    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    cols = []
    for file in files_train:
        # col 0 is the index in the file
        trainset = pd.read_csv(file, index_col=0, dtype=np.float32)
        if len(cols) == 0:
            cols = trainset.columns
        scaler.partial_fit(trainset[col_names_scaling])

    del trainset
    cols_keep = list(set(cols) - set(col_names_scaling))

    for file in files_train:
        trainset = pd.read_csv(file, dtype=np.float32)
        train_scaled = scaler.transform(trainset[col_names_scaling])
        train_changed = pd.DataFrame(train_scaled, columns=col_names_scaling)
        train_unchanged = trainset[cols_keep]
        trainset_final = pd.concat((train_changed, train_unchanged),
                                   axis=1)
        trainset_final = trainset_final[cols]
        print("train:", trainset_final.shape)  # trainset shape
        file_csv = file_folder + file
        trainset.to_csv(file_csv, index=False)
        len_tail = len('.csv')
        file_tfr = file_folder + file[:-1 * len_tail] + '.tfrecords'
        make_tfrecords(trainset_final, file_tfr)

    for file in files_test:
        testset = pd.read_csv(file, dtype=np.float32)
        test_scaled = scaler.transform(testset[col_names_scaling])
        test_changed = pd.DataFrame(test_scaled, columns=col_names_scaling)
        test_unchanged = testset[cols_keep]
        testset_final = pd.concat((test_changed, test_unchanged), axis=1)
        testset_final = testset_final[cols]
        print("test:", testset_final.shape)
        file_csv = file_folder + file
        testset.to_csv(file_csv, index=False)
        len_tail = len('.csv')
        file_tfr = file_folder + file[:-1 * len_tail] + '.tfrecords'
        make_tfrecords(testset_final, file_tfr)


# Save the file in tf format
# 将文件保存成tf格式
def make_tfrecords(dataset, file_to_save):
    try:
        data = dataset.values
    except:
        data = dataset
    with tf.compat.v1.python_io.TFRecordWriter(file_to_save) as writer:  # python_io在tfv1中
        for rows in data:
            features, label_10, label_2 = rows[:-2], rows[-2], rows[-1]
            feature = {'features': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                       'label_2': tf.train.Feature(float_list=tf.train.FloatList(value=[label_2])),
                       'label_10': tf.train.Feature(float_list=tf.train.FloatList(value=[label_10]))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def next_batch(filename, batch_size):
    len_feature = 202  # 特征数（不包含标签）。 Number of features (not including tags)
    len_label = 1  # 标签长度。 The length of the label

    def read_data(examples):
        features = {"features": tf.io.FixedLenFeature([len_feature], tf.float32),
                    "label_2": tf.io.FixedLenFeature([len_label], tf.float32),
                    "label_10": tf.io.FixedLenFeature([len_label], tf.float32)}
        parsed_features = tf.io.parse_single_example(examples, features)
        return parsed_features['features'], parsed_features['label_2'], \
               parsed_features['label_10']

    data = tf.data.TFRecordDataset(filename)
    data = data.map(read_data)
    data = data.batch(batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(data)
    next_data, next_label_2, next_label_10 = iterator.get_next()

    return next_data, next_label_10, next_label_2


# Integrate the four separate data sets
# 将分开的4个数据集整合到一起
def make_whole_datasets(tfrecords_train, num_train_example, tfrecords_test,
                        num_test_example):
    with tf.compat.v1.Session() as sess:
        data_test, label_10_test, label_2_test = next_batch(tfrecords_test, num_test_example)
        data, label_10, label_2 = sess.run([data_test, label_10_test, label_2_test])
    dataset = np.concatenate([data, label_10, label_2], axis=1)
    print("test:", dataset.shape)
    make_tfrecords(dataset, 'normalized/test.tfrecords')


if __name__ == '__main__':
    file_folder = 'F:/数据挖掘/SGM-CNN/UNSW_NB15/'  # 读取的原始文件所在的位置。 The location where the original file was read
    col_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
                 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin',
                 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
                 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt',
                 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips',
                 'ct_state_ttl', 'ct_flw', 'is_ftp', 'ct_ftp', 'ct_srv_src',
                 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport',
                 'ct_dst_sport', 'ct_dst_src', 'label_10', 'label_2']  # 特证名（列名）。 listed name

    cols_to_drop = ['srcip', 'dstip', 'stime', 'ltime', 'sport', 'dsport']
    cols_nominal = ['proto', 'service', 'state']  # 名词性特征。Nominal features

    files = [file_folder + 'UNSW-NB15_' + str(i + 1) + '.csv' for i in range(4)]
    # dataset = combine_dataset(files, col_names)
    file_folder = 'normalized/'  # 数据标准化后存放的文件夹。A folder where data is stored after standardization
    files_train = [file_folder + str(x + 1) + '_train.tfrecords' for x in range(4)]
    files_test = [file_folder + str(x + 1) + '_test.tfrecords' for x in range(4)]
    num_train_example = 2032035  # trainset size
    num_test_example = 508012  # testset size
    make_whole_datasets(files_train, num_train_example, files_test, num_test_example)
