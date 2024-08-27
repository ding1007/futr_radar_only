import pickle
with  open('data/nuscenes_futr_infos_train.pkl', 'rb') as f:
    data=pickle.load(f)
print(data)