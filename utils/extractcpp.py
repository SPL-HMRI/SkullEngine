import os

def extract_cpp(file):
  if file.endswith('.csv'):
    landmark_all = pd.read_csv(file)
    # landmark_all = pd.read_csv('/home/admin/test/data_v1_ldmk/case_17_cbct_patient.csv')
    landmark_list = landmark_all['name'].tolist()
    landmark = landmark_all[landmark_all['name'] == 'COR-R']
    landmark = [landmark['x'].values[0], landmark['y'].values[0], landmark['z'].values[0]]
  elif file.endswith('.xlsx'):
    landmark_all = pd.read_excel(file)
    landmark_list = landmark_all['Landmark Name'].tolist()
    landmark_offset = landmark_all[landmark_all['Landmark Name'] == 'Landmark_Offset']
    landmark_offset = [landmark_offset['Original_X'].values[0], landmark_offset['Original_Y'].values[0],
                       landmark_offset['Original_Z'].values[0]]
    landmark_temp = landmark_all[landmark_all['Landmark Name'] == 'COR-R']
    landmark_temp = [landmark_temp['Original_X'].values[0], landmark_temp['Original_Y'].values[0],
                     landmark_temp['Original_Z'].values[0]]
    landmark = [landmark_offset[i] + landmark_temp[i] for i in range(len(landmark_offset))]
    print('Coordinate: {}'.format(landmark))
  elif lm_path.file('.txt'):
    lines_all = pd.read_csv(file, lineterminator='1>') #, names=['name', 'x', 'y', 'z'])
    lines_list = landmark_all['name'].tolist()
    landmark = landmark_all[landmark_all['name'] == 'ION-R']
    landmark = [landmark['x'].values[0], landmark['y'].values[0], landmark['z'].values[0]]


if __name__ == '__main__':
  #
  path = 'E:\AAProgram\split_2022-current\Output-Build.txt'

  extract_cpp(path)