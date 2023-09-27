import torch
import numpy as np
import time
import scipy.io as sio

def ber(Recover_X,Ideal_X):
    batchsize, S, F = Recover_X.shape[0], Recover_X.shape[1], Recover_X.shape[2]
    Ideal_X = Ideal_X.float()
    Recover_X_id=(torch.sign(torch.stack((-Recover_X.real,-Recover_X.imag),dim = -1)) + 1)/2
    Recover_X_id[:,0,::2,:] = Ideal_X[:,0,::2,:]
    ber = (Ideal_X != Recover_X_id).sum()/(batchsize*(S-0.5)*F*2)
    return ber


class Tester():
    def __init__(self, args, model):
        self.args = args
        self.layers = args.layers
        self.model = model
        self.ckp_dir = args.ckp_dir
        self.sigma2 = 1.0


    def test(self):
        self.model.load_model(self.args)
        NumSample = 22500
        S = 12
        F = 24
        Received_Y = np.empty([NumSample,S,F,4,2])
        Hls = np.empty([NumSample,S,F,4,2])
        Ideal_H = np.empty([NumSample,S,F,4,2])
        Ideal_X = np.empty([NumSample,S,F,2])
        Transmit_X = np.empty([NumSample,S,F,2])


        db = [-10,-5,0,5,10,15,20,25,30]
        #db = -5  

        for i in range(len(db)):
            data = sio.loadmat('E:/paperwithcode/MIMO_JCESD-main/GenerateH/MIMO/CDL-A{}dB_{}Hz_R_{}.mat'.format(db[i],self.args.doppler,4))
            Received_Y[i:NumSample:len(db),:,:,:,:] = data['Received_Y']
            Hls[i:NumSample:len(db),:,:,:,:] = data['Hls']
            Ideal_H[i:NumSample:len(db   ),:,:,:,:] = data['Ideal_H']
            Ideal_X[i:NumSample:len(db),:,:,:]= data['Ideal_X']
            Transmit_X[i:NumSample:len(db),:,:,:]= data['Transmit_X']
            
        
        id = np.arange(0,NumSample,1)
        id = np.argwhere(id % (NumSample/1000) < int(NumSample/1000*0.4))
        Received_Y = torch.from_numpy(np.delete(Received_Y,id,0))
        Hls = torch.from_numpy(np.delete(Hls,id,0))
        Ideal_H = torch.from_numpy(np.delete(Ideal_H,id,0))
        Ideal_X = torch.from_numpy(np.delete(Ideal_X,id,0))
        Transmit_X = torch.from_numpy(np.delete(Transmit_X,id,0))

        self.model.net.eval()
        for j in range(len(db)):
            id_test = torch.arange(int(NumSample*0.6*0.8)+j,int(NumSample*0.6)+j,len(db)).long()
            Received_Y_test = Received_Y[id_test,:,:,:,:].cuda()
            Hls_test = Hls[id_test,:,:,:,:].cuda()
            Ideal_H_test = Ideal_H[id_test,:,:,:,:].cuda()
            Ideal_X_test = Ideal_X[id_test,:,:,:].cuda()
            Transmit_X_test = Transmit_X[id_test,:,:,:].cuda()
            test_size = int(NumSample*0.12/len(db))

            #Received_Y_test = shift(Received_Y_test,ff,0)
            #Hls_test = cdiv(Received_Y_test,Transmit_X_test[:,:,:,None,:].repeat(1,1,1,4,1))

            batchsize = 100
            ber_vl = 0.0
            for i in range(int(test_size//batchsize)):
                batch_idx = torch.arange(batchsize*i,batchsize*(i+1),1)
                Received_Y_batch = Received_Y_test[batch_idx,:,:,:,:].cuda()
                Hls_batch = Hls_test[batch_idx,:,:,:,:].cuda()
                Ideal_H_batch = Ideal_H_test[batch_idx,:,:,:,:].cuda()
                Ideal_X_batch = Ideal_X_test[batch_idx,:,:,:].cuda()
                Transmit_X_batch = Transmit_X_test[batch_idx,:,:,:].cuda()

                H_full_batch,Recover_X_batch,sigma2 = self.model.net(Received_Y_batch,  Hls_batch, Transmit_X_batch)
                ber_vl = ber_vl + ber(Recover_X_batch,Ideal_X_batch)
            ber_vl = ber_vl/(int(test_size//batchsize))
            print('{:d} BER = {:.6f}'.format(j,ber_vl))