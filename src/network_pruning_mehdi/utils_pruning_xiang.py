import numpy as np
import torch

def backward_selection_joint(W, Wbar, XTX, XTY, XX, lam_comb, lam_joint, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    
    
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
    num_prune = np.sum(prune_list)
    
    XTX_inv = np.zeros((num_cout, totp, totp))
    XTX_tot = np.zeros((num_cout, totp, totp))
    for iout in range(num_cout):
        XTX_tot[iout, :, :] = XTX + lam_comb * XX[:,:,iout].T @ XX[:,:,iout] + lam_comb * lam_joint * np.eye(totp)
        XTX_inv[iout, prune_list2[:, np.newaxis], prune_list2] = np.linalg.inv(XTX_tot[iout, prune_list2[:, np.newaxis], prune_list2])
        W[:, iout] = XTX_inv[iout,:,:] @ (XTY[:, iout] + lam_comb * XX[:,:,iout].T @ XX[:,:,iout] @ Wbar[:, iout] + lam_comb * lam_joint * Wbar[:, iout])
    
    
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        obj_mat = np.zeros_like(W)
        for iout in range(num_cout):
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2*ksize:(i2+1)*ksize, iout] = np.linalg.inv(XTX_inv[iout, i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]) @ W[i2*ksize:(i2+1)*ksize,iout]
        
        obj_cha = W * obj_mat    
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)  
        idx = np.argsort(obj_sum + 1e20*(prune_list) )
       
        for iout in range(num_cout):
            W[:,iout] -= XTX_inv[iout,:,idx[0]*ksize:(idx[0]+1)*ksize] @ np.linalg.inv(XTX_inv[iout,idx[0]*ksize:(idx[0]+1)*ksize,idx[0]*ksize:(idx[0]+1)*ksize]) @ W[idx[0]*ksize:(idx[0]+1)*ksize,iout]
            
        W[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        
        prune_list[idx[0]] = True
        prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
        
        for iout in range(num_cout):
    
            
            XTX_inv[iout,prune_list2[:, np.newaxis], prune_list2] += (XTX_inv[iout,prune_list2[:, np.newaxis], prune_list2]@XTX_tot[iout,prune_list2,idx[0]*ksize:(idx[0]+1)*ksize])@np.linalg.inv(
                                                  np.eye(ksize)-XTX_inv[iout,idx[0]*ksize:(idx[0]+1)*ksize,prune_list2].T@XTX_tot[iout,prune_list2,idx[0]*ksize:(idx[0]+1)*ksize]  )@XTX_inv[iout,idx[0]*ksize:(idx[0]+1)*ksize,prune_list2].T
        
        
            XTX_inv[iout,idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
            XTX_inv[iout,:,idx[0]*ksize:(idx[0]+1)*ksize] = 0
        
      
    W_sol = np.zeros_like(W)
    for iout in range(num_cout):
        nzi = np.nonzero(W[:,iout])[0]
        XTX_sub = (XTX + lam_comb * XX[:,:,iout].T @ XX[:,:,iout] + lam_comb * lam_joint * np.eye(totp))[np.ix_(nzi,nzi)]
        XTY_sub = (XTY[:, iout] + lam_comb * XX[:,:,iout].T @ XX[:,:,iout] @ Wbar[:, iout] + lam_comb * lam_joint * Wbar[:, iout])[nzi]
        W_sol[nzi,iout] = np.linalg.inv(XTX_sub)@ XTY_sub
        
    W_obj = 0
    for iout in range(num_cout):
        W_obj += np.sum((1/2)*W_sol[:,iout] * (XTX_tot[iout,:,:]@W_sol[:,iout]))
        W_obj -= np.sum(W_sol[:,iout] * (XTY[:, iout] + lam_comb * XX[:,:,iout].T @ XX[:,:,iout] @ Wbar[:, iout] + lam_comb * lam_joint * Wbar[:, iout]))
    return W_sol, W_obj

def evaluate_obj(W, Wbar, XTX, XTY, XX, lam_joint):
    
    W_obj = 0
    totp, num_cout = W.shape
    XTX_tot = np.zeros((num_cout, totp, totp))
    for iout in range(num_cout):
        XTX_tot[iout, :, :] = XX[:,:,iout].T @ XX[:,:,iout] + lam_joint * np.eye(totp)
        
    for iout in range(num_cout):
        W_obj += np.sum((1/2)*W[:,iout] * (XTX_tot[iout,:,:]@W[:,iout]))
        W_obj -= np.sum(W[:,iout] * (XX[:,:,iout].T @ XX[:,:,iout] @ Wbar[:, iout] + lam_joint * Wbar[:, iout]))
    
    return np.sum( -W * XTY + (1/2) * W * (XTX@W) ) , W_obj

def back_solve_regu(w_bar, W, XX, lambda2, XTX2, XTY2, lambda3 = 0):
    
    p, m = W.shape
    W_sol = np.zeros_like(W)
    for k in range(m):
        nzi = np.nonzero(W[:,k])[0]
        
        XTX = XX[:,:,k].T @ XX[:,:,k]
        
        
        #print(XX.shape, w_bar.shape)
        
        XTY = XTX @ w_bar[:,k]
        
        XTX += lambda3 * np.eye(XTX.shape[0])
        XTY += lambda3 * w_bar[:,k]
        
        XTX_sub = XTX[np.ix_(nzi,nzi)]
        XTY_sub = XTY[nzi]
        
        XTX2_sub = XTX2[np.ix_(nzi,nzi)]
        XTY2_sub = XTY2[nzi,k]
        
        W_sol[nzi,k] = np.linalg.solve(XTX_sub + lambda2 * XTX2_sub, XTY_sub + lambda2 * XTY2_sub)
    return W_sol

"""
#
# only prune certain layers in residual block
#
for ssi in range(len(spar_list)):
    
    for pai in range(len(lam_list)): 
        
        lam_comb = lam_list[pai]
        sparsity = spar_list[ssi]
        i_w = 0
        w_prunedL = np.copy(w_bar)
    
        # Hessian and X^\topy placeholder
        hess_list = []
        xty_list = []
        for si in range(len(size_list)-1):
            sum_w = np.prod(size_list[si][1:])
            hess_list.append(np.zeros((sum_w,sum_w)))
            xty_list.append(np.zeros((sum_w,size_list[si][0])))
        
        for si in range(len(size_list)-1):
    
            count = np.prod(size_list[si])
            if si not in [2,4,6,8,10,12,14,16,18]:
                i_w += count
                #print("Layer "+str(si)+" finished")
                continue
        
            torch.manual_seed(ssi * 20 + si)
            torch.cuda.manual_seed(ssi * 20 + si)
            torch.cuda.manual_seed_all(ssi * 20 + si)
            np.random.seed(ssi * 20 + si)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=10,pin_memory=True)
            # collect data
            for ci, batch in enumerate(train_dataloader):
    
                xdata, ydata = batch
                xdata = xdata.to("cuda")
                model_new = copy.deepcopy(model)
                model_new2 = copy.deepcopy(model)
            
                set_pvec(w_prunedL, model_new, modules_to_prune,"cuda")
                set_pvec(w_bar, model_new2, modules_to_prune,"cuda")
            
                foo(model_new)
    
                input_buffer = []
                output_buffer = []
                model_new(xdata)
                save1T = copy.deepcopy([np.array(xx.to("cpu").detach().numpy()) for xx in input_buffer])
                save2T = copy.deepcopy([np.array(xx.to("cpu").detach().numpy()) for xx in output_buffer])
            
                foo(model_new2)
    
                input_buffer = []
                output_buffer = []
                model_new2(xdata)
                save1T2 = copy.deepcopy([np.array(xx.to("cpu").detach().numpy()) for xx in input_buffer])
                save2T2 = copy.deepcopy([np.array(xx.to("cpu").detach().numpy()) for xx in output_buffer])
    

                w_cur = w_prunedL[i_w:i_w+count]
                if si == 7 or si == 13:
                    stride = 2
                else:
                    stride = 1
            
                xHess, xty = my_convo(w_cur.reshape(size_list[si]), save1T[si][0], save2T2[si], stride)
                hess_list[si] += xHess
                xty_list[si] += xty
        
                if (ci + 1) % num_sample == 0:
                    break
                    
            torch.manual_seed(ssi * 20 + si)
            torch.cuda.manual_seed(ssi * 20 + si)
            torch.cuda.manual_seed_all(ssi * 20 + si)
            np.random.seed(ssi * 20 + si)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=10,pin_memory=True)
            

            Xgrad = torch.zeros((num_grad, count), device='cpu')
            for ci, batch in enumerate(train_dataloader):
    
                xdata, ydata = batch
                xdata = xdata.to("cuda")
                ydata = ydata.to("cuda")
                model_new = copy.deepcopy(model)
                set_pvec(w_prunedL, model_new, modules_to_prune,"cuda")
                loss = criterion(model_new(xdata), ydata)
                loss.backward()
                Xgrad[ci] = get_gvec(model_new, modules_to_prune).to('cpu')[i_w:i_w+count]
                zero_grads(model_new)
                if (ci + 1) % num_grad == 0:
                    break
            
            ## XX: num_grad * size of neuron/channel * num_neurons
            XX = Xgrad.numpy().astype(np.float64).reshape(num_grad,size_list[si][0],-1).swapaxes(1,2)
            
            param_size = size_list[si]
            w_cur = np.copy(w_bar[i_w:i_w+count])
            stt = time.time()
            
            w_input = (w_cur.reshape(-1).reshape(param_size[0],-1)).T
           
            #hess_list[si], xty_list[si]: hessian and linear term for each channel / neuron
            w_sol, w_obj = backward_selection_joint(np.copy(w_input),np.copy(w_input), 
                            hess_list[si], xty_list[si], XX, lam_comb, 1e-2, param_size[0], int(param_size[1] * (1-sparsity)))
                
            w_obj1, w_obj2 = evaluate_obj(np.copy(w_sol),np.copy(w_input), hess_list[si], xty_list[si], XX, 1e-2)
            
            tab_list3[ssi,pai,si] = w_obj1
            tab_list4[ssi,pai,si] = w_obj2
        
                
            w_prunedL[i_w:i_w+count] =  (w_sol.T).reshape(-1)   
            i_w += count
            print(si,end=" ")
            
    
        w_prunedL[i_w:] = np.copy(w_bar[i_w:])
    
        tab_list[ssi,pai] = getacc(w_prunedL)
        
        model_new3 = copy.deepcopy(model)
        set_pvec(w_prunedL, model_new3, modules_to_prune,"cuda")
        propagate_sparsity(model_new3,train_dataloader,samples=100)
        w_prunedL2 = get_pvec(model_new3, modules_to_prune).to('cpu').numpy().astype(np.float64)
        tab_list2[ssi,pai] = len(np.where(w_prunedL2)[0])
        print("Finished:",ssi,pai)
"""