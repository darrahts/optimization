import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cross_validation
import numpy as np
import random as rd

Data = pd.read_csv("spambase.csv")

Data = Data.sample(frac=1)

Xold = Data.drop(["Y"],axis=1)
Y = pd.DataFrame(Data,columns=["Y"]).values


norm = preprocessing.MinMaxScaler()
X = norm.fit_transform(Xold)


pca = PCA(n_components=30)
X = pca.fit_transform(X)

Var = pca.explained_variance_ratio_

Exp_Var = 0

for i in Var:
    Exp_Var += i

print("\n")
print(Exp_Var)




Cnt1 = len(X)
print()
print("# of Obsrvations:",Cnt1)

      
p_c = 1 # prob. of crossover
p_m = 0.2 # prob. of mutation
pop = 40
gen = 20
kfold = 3



XY0 = np.array([1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,1,0,1,0])


n_list = np.empty((0,len(XY0)))


for i in range(pop):
    rd.shuffle(XY0)
    n_list = np.vstack((n_list,XY0))


# Calculate the FV
    
# X = C
a_X = 10
b_X = 1000
l_X = (len(XY0)/2)


# Y = gamma
a_Y = 0.05
b_Y = 0.99
l_Y = (len(XY0)/2)


Precision_X = (b_X - a_X)/((2**l_X)-1)
Precision_Y = (b_Y - a_Y)/((2**l_Y)-1)


z = 0
t = 1
X0_num_Sum = 0

for i in range(len(XY0)//2):
    X0_num = XY0[-t]*2**z
    X0_num_Sum += X0_num
    t = t+1
    z = z+1
    
p = 0
u = 1+(len(XY0)//2)
Y0_num_Sum = 0

for i in range(len(XY0)//2):
    Y0_num = XY0[-u]*2**p
    Y0_num_Sum += Y0_num
    u = u+1
    p = p+1


Decoded_X = (X0_num_Sum*Precision_X)+a_X
Decoded_Y = (Y0_num_Sum*Precision_Y)+a_Y

print()
print("Decoded_X:",Decoded_X)
print("Decoded_Y:",Decoded_Y)


Final_Best_in_Generation_X = []
Worst_Best_in_Generation_X = []

One_Final_Guy = np.empty((0,len(XY0)+2))
One_Final_Guy_Final = []

Min_for_all_Generations_for_Mut_1 = np.empty((0,len(XY0)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(XY0)+1))

Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(XY0)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(XY0)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(XY0)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(XY0)+2))


Generation = 1


for i in range(gen):
    
    
    New_Population = np.empty((0,len(XY0)))
    
    All_in_Generation_X_1 = np.empty((0,len(XY0)+1))
    All_in_Generation_X_2 = np.empty((0,len(XY0)+1))
    
    Min_in_Generation_X_1= []
    Min_in_Generation_X_2 = []
    
    
    Save_Best_in_Generation_X = np.empty((0,len(XY0)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    
    print()
    print("GENERATION: #:",Generation)
        
    Family = 1
    
    for j in range(int(pop/2)):
        
        print()
        print("Family: #:",Family)
        
        # Tour. Selection
        
        Parents = np.empty((0,len(XY0)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Warrior_1_index = np.random.randint(0,len(n_list))
            Warrior_2_index = np.random.randint(0,len(n_list))
            Warrior_3_index = np.random.randint(0,len(n_list))
            
            while Warrior_1_index==Warrior_2_index:
                Warrior_1_index = np.random.randint(0,len(n_list))
            while Warrior_2_index==Warrior_3_index:
                Warrior_3_index = np.random.randint(0,len(n_list))
            while Warrior_1_index==Warrior_3_index:
                Warrior_3_index = np.random.randint(0,len(n_list))



            Warrior_1 = n_list[Warrior_1_index]
            Warrior_2 = n_list[Warrior_2_index]
            Warrior_3 = n_list[Warrior_3_index]
            
            Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
            
            
            # Warrior 1
            
            z = 0
            t = 1
            X0_num_Sum_W1 = 0
            
            for i in range(len(XY0)//2):
                X0_num_W1 = Warrior_1[-t]*2**z
                X0_num_Sum_W1 += X0_num_W1
                t = t+1
                z = z+1
            
            Y0_num_Sum_W1 = 0
            p = 0
            u = 1+(len(XY0)//2)
            Y0_num_Sum = 0
            
            for i in range(len(XY0)//2):
                Y0_num_W1 = Warrior_1[-u]*2**p
                Y0_num_Sum_W1 += Y0_num_W1
                u = u+1
                p = p+1
            
            
            Decoded_X_W1 = (X0_num_Sum_W1*Precision_X)+a_X
            Decoded_Y_W1 = (Y0_num_Sum_W1*Precision_Y)+a_Y
            '''
            print()
            print("Decoded_X_W1:",Decoded_X_W1)
            print("Decoded_Y_W1:",Decoded_Y_W1)
            '''
            P_1 = 0
            
            kf = cross_validation.KFold(Cnt1,n_folds=kfold)
            
            for train_index,test_index in kf:
                X_train,X_test = X[train_index],X[test_index]
                Y_train,Y_test = Y[train_index],Y[test_index]
                
                model1 = svm.SVC(kernel="rbf",C=Decoded_X_W1,gamma=Decoded_Y_W1)
                model1.fit(X_train,Y_train)
                PL1 = model1.predict(X_test)
                
                AC1 = model1.score(X_test,Y_test)
                
                OF_So_Far_1 = 1-(AC1)
                
                P_1 += OF_So_Far_1
                
            OF_So_Far_W1 = P_1/kfold
            
            
            
            # Warrior 2
            
            z = 0
            t = 1
            X0_num_Sum_W2 = 0
            
            for i in range(len(XY0)//2):
                X0_num_W1 = Warrior_1[-t]*2**z
                X0_num_Sum_W1 += X0_num_W1
                t = t+1
                z = z+1
            
            Y0_num_Sum_W2 = 0
            p = 0
            u = 1+(len(XY0)//2)
            Y0_num_Sum = 0
            
            for i in range(len(XY0)//2):
                Y0_num_W2 = Warrior_2[-u]*2**p
                Y0_num_Sum_W2 += Y0_num_W2
                u = u+1
                p = p+1
            
            
            Decoded_X_W2 = (X0_num_Sum_W2*Precision_X)+a_X
            Decoded_Y_W2 = (Y0_num_Sum_W2*Precision_Y)+a_Y
            '''
            print()
            print("Decoded_X_W2:",Decoded_X_W2)
            print("Decoded_Y_W2:",Decoded_Y_W2)
            '''
            P_1 = 0
            
            kf = cross_validation.KFold(Cnt1,n_folds=kfold)
            
            for train_index,test_index in kf:
                X_train,X_test = X[train_index],X[test_index]
                Y_train,Y_test = Y[train_index],Y[test_index]
                
                model1 = svm.SVC(kernel="rbf",C=Decoded_X_W2,gamma=Decoded_Y_W2)
                model1.fit(X_train,Y_train)
                PL1 = model1.predict(X_test)
                
                AC1 = model1.score(X_test,Y_test)
                
                OF_So_Far_2 = 1-(AC1)
                
                P_1 += OF_So_Far_2
                
            OF_So_Far_W2 = P_1/kfold
            
            # Warrior 3
            
            z = 0
            t = 1
            X0_num_Sum_W3 = 0
            
            for i in range(len(XY0)//2):
                X0_num_W1 = Warrior_3[-t]*2**z
                X0_num_Sum_W3 += X0_num_W1
                t = t+1
                z = z+1
            
            Y0_num_Sum_W3 = 0
            p = 0
            u = 1+(len(XY0)//2)
            Y0_num_Sum = 0
            
            for i in range(len(XY0)//2):
                Y0_num_W3 = Warrior_3[-u]*2**p
                Y0_num_Sum_W3 += Y0_num_W1
                u = u+1
                p = p+1
            
            
            Decoded_X_W3 = (X0_num_Sum_W1*Precision_X)+a_X
            Decoded_Y_W3 = (Y0_num_Sum_W1*Precision_Y)+a_Y
            '''
            print()
            print("Decoded_X_W3:",Decoded_X_W3)
            print("Decoded_Y_W3:",Decoded_Y_W3)
            '''
            P_1 = 0
            
            kf = cross_validation.KFold(Cnt1,n_folds=kfold)
            
            for train_index,test_index in kf:
                X_train,X_test = X[train_index],X[test_index]
                Y_train,Y_test = Y[train_index],Y[test_index]
                
                model1 = svm.SVC(kernel="rbf",C=Decoded_X_W3,gamma=Decoded_Y_W3)
                model1.fit(X_train,Y_train)
                PL1 = model1.predict(X_test)
                
                AC1 = model1.score(X_test,Y_test)
                
                OF_So_Far_1 = 1-(AC1)
                
                P_1 += OF_So_Far_1
                
            OF_So_Far_W3 = P_1/kfold

            
            Prize_Warrior_1 = OF_So_Far_W1
            Prize_Warrior_2 = OF_So_Far_W2
            Prize_Warrior_3 = OF_So_Far_W3
            
            
            if Prize_Warrior_1 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_1
                Prize = Prize_Warrior_1

            if Prize_Warrior_2 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_2
                Prize = Prize_Warrior_2
            
            if Prize_Warrior_3 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_3
                Prize = Prize_Warrior_3


            Parents = np.vstack((Parents,Winner))
        '''
        print(Parents)
        '''
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        # Crossover
        
        Child_1 = np.empty((0,len(XY0)))
        Child_2 = np.empty((0,len(XY0)))
        
        
        
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c:
            
            
            Cr_1 = np.random.randint(0,len(XY0))
            Cr_2 = np.random.randint(0,len(XY0))
            
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(0,len(XY0))
                
            if Cr_1 < Cr_2:
                
                Med_Seg_1 = Parent_1[Cr_1:Cr_2+1]
                Med_Seg_2 = Parent_2[Cr_1:Cr_2+1]
                
                First_Seg_1 = Parent_1[:Cr_1]
                Sec_Seg_1 = Parent_1[Cr_2+1:]
                
                First_Seg_2 = Parent_2[:Cr_1]
                Sec_Seg_2 = Parent_2[Cr_2+1:]
                
                Child_1 = np.concatenate((First_Seg_1,Med_Seg_2,Sec_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Med_Seg_1,Sec_Seg_2))
                
            else:
                
                Med_Seg_1 = Parent_1[Cr_2:Cr_1+1]
                Med_Seg_2 = Parent_2[Cr_2:Cr_1+1]
                
                First_Seg_1 = Parent_1[:Cr_2]
                Sec_Seg_1 = Parent_1[Cr_1+1:]
                
                First_Seg_2 = Parent_2[:Cr_2]
                Sec_Seg_2 = Parent_2[Cr_1+1:]
                
                Child_1 = np.concatenate((First_Seg_1,Med_Seg_2,Sec_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Med_Seg_1,Sec_Seg_2))
        
        else:
            Child_1 = Parent_1
            Child_2 = Parent_2
            
        '''
        print()
        print("Child_1:",Child_1)
        print("Child_2:",Child_2)
        '''
        
        Mutated_Child_1 = []
        
        t = 0
        
        for i in Child_1:
            
            Ran_Mut_1 = np.random.rand()
            
            if Ran_Mut_1 < p_m:
                
                if Child_1[t] == 0:
                    Child_1[t] = 1
                else:
                    Child_1[t] = 0
                t = t+1
                
                Mutated_Child_1 = Child_1
                
            else:
                Mutated_Child_1 = Child_1
                
        
        Mutated_Child_2 = []
        
        t = 0
        
        for i in Child_2:
            
            Ran_Mut_2 = np.random.rand()
            
            if Ran_Mut_2 < p_m:
                
                if Child_2[t] == 0:
                    Child_2[t] = 1
                else:
                    Child_2[t] = 0
                t = t+1
                
                Mutated_Child_2 = Child_2
                
            else:
                Mutated_Child_2 = Child_2
        '''
        print()
        print("Mutated_Child_1:",Mutated_Child_1)
        print("Mutated_Child_2:",Mutated_Child_2)
        '''
        
        # For mutated child_1
        
        X0_num_Sum_MC_1 = 0
        Y0_num_Sum_MC_1 = 0
        
        z = 0
        t = 1
        
        
        for i in range(len(XY0)//2):
            X0_num_M1 = Mutated_Child_1[-t]*2**z
            X0_num_Sum_MC_1 += X0_num_M1
            t = t+1
            z = z+1
        
        p = 0
        u = 1+(len(XY0)//2)
        
        for i in range(len(XY0)//2):
            Y0_num_M1 = Mutated_Child_1[-u]*2**p
            Y0_num_Sum_MC_1 += Y0_num_M1
            u = u+1
            p = p+1
        
        
        Decoded_X_M1 = (X0_num_Sum_MC_1*Precision_X)+a_X
        Decoded_Y_M1 = (Y0_num_Sum_MC_1*Precision_Y)+a_Y
        '''
        print()
        print("Decoded_X_M1:",Decoded_X_M1)
        print("Decoded_Y_M1:",Decoded_Y_M1)
        '''
        P_1 = 0
        
        kf = cross_validation.KFold(Cnt1,n_folds=kfold)
        
        for train_index,test_index in kf:
            X_train,X_test = X[train_index],X[test_index]
            Y_train,Y_test = Y[train_index],Y[test_index]
            
            model1 = svm.SVC(kernel="rbf",C=Decoded_X_M1,gamma=Decoded_Y_M1)
            model1.fit(X_train,Y_train)
            PL1 = model1.predict(X_test)
            
            AC1 = model1.score(X_test,Y_test)
            
            OF_So_Far_1 = 1-(AC1)
            
            P_1 += OF_So_Far_1
            
        OF_So_Far_M1 = P_1/kfold


        # For mutated child_2
        
        X0_num_Sum_MC_2 = 0
        Y0_num_Sum_MC_2 = 0
        
        z = 0
        t = 1
        
        
        for i in range(len(XY0)//2):
            X0_num_M2 = Mutated_Child_2[-t]*2**z
            X0_num_Sum_MC_2 += X0_num_M2
            t = t+1
            z = z+1
        
        p = 0
        u = 1+(len(XY0)//2)
        
        for i in range(len(XY0)//2):
            Y0_num_M2 = Mutated_Child_2[-u]*2**p
            Y0_num_Sum_MC_2 += Y0_num_M2
            u = u+1
            p = p+1
        
        
        Decoded_X_M2 = (X0_num_Sum_MC_2*Precision_X)+a_X
        Decoded_Y_M2 = (Y0_num_Sum_MC_2*Precision_Y)+a_Y
        '''
        print()
        print("Decoded_X_M2:",Decoded_X_M2)
        print("Decoded_Y_M2:",Decoded_Y_M2)
        '''
        P_1 = 0
        
        kf = cross_validation.KFold(Cnt1,n_folds=kfold)
        
        for train_index,test_index in kf:
            X_train,X_test = X[train_index],X[test_index]
            Y_train,Y_test = Y[train_index],Y[test_index]
            
            model1 = svm.SVC(kernel="rbf",C=Decoded_X_M2,gamma=Decoded_Y_M2)
            model1.fit(X_train,Y_train)
            PL1 = model1.predict(X_test)
            
            AC1 = model1.score(X_test,Y_test)
            
            OF_So_Far_2 = 1-(AC1)
            
            P_1 += OF_So_Far_2
            
        OF_So_Far_M2 = P_1/kfold
        
        
        print()
        print("FV for Mutated Child #1 at Gen",Generation,":",OF_So_Far_M1)
        print("FV for Mutated Child #2 at Gen",Generation,":",OF_So_Far_M2)


        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        
        All_in_Generation_X_1_1 = np.column_stack((OF_So_Far_M1,All_in_Generation_X_1_1_temp))
        
        All_in_Generation_X_2_2_temp = Mutated_Child_2[np.newaxis]
        
        All_in_Generation_X_2_2 = np.column_stack((OF_So_Far_M2,All_in_Generation_X_2_2_temp))
        
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_2))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))


        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        
        R_1 = []
        t = 0
        for i in All_in_Generation_X_1:
            
            if(All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                R_1 = All_in_Generation_X_1[t,:]
            t = t+1
            
        
        Min_in_Generation_X_1 = R_1[np.newaxis]
        
        
        R_2 = []
        t = 0
        for i in All_in_Generation_X_2:
            
            if(All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                R_2 = All_in_Generation_X_2[t,:]
            t = t+1
            
        
        Min_in_Generation_X_2 = R_2[np.newaxis]
        
        
        Family = Family+1
        
        
    t = 0
    R_11 = []
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            R_11 = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    Final_Best_in_Generation_X = R_11[np.newaxis]
    
    
    t = 0
    R_22 = []
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            R_22 = Save_Best_in_Generation_X[t,:1]
        t = t+1
            
    Worst_Best_in_Generation_X = R_22[np.newaxis]
    
    
    Darwin_Guy = Final_Best_in_Generation_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
    
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    Best_1 = np.where((New_Population==Darwin_Guy))
    Worst_1 = np.where((New_Population==Not_So_Darwin_Guy))
    
    
    New_Population[Worst_1] = Darwin_Guy
    
    n_list = New_Population

    
    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,
                                                   Min_in_Generation_X_1))
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,
                                                   Min_in_Generation_X_2))
    
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1,0,Generation)
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2,0,Generation)
    
    
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,
                                                       Min_for_all_Generations_for_Mut_1_1))
    
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,
                                                       Min_for_all_Generations_for_Mut_2_2))
    
    
    Generation = Generation+1
    
    
    
One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,
                           Min_for_all_Generations_for_Mut_2_2_2))


t = 0
Final_Here = []

for i in One_Final_Guy:
    if(One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
        Final_Here = One_Final_Guy[t,:]
    t = t+1

One_Final_Guy_Final = Final_Here[np.newaxis]

print()
print("Min in all Generations:",One_Final_Guy_Final)

print()
print("Final Solution:",One_Final_Guy_Final[:,2:])
print("Highest Accuracy:",(1-One_Final_Guy_Final[:,1]))

XY0_Encoded_After = One_Final_Guy_Final[:,2:][0]


z = 0
t = 1
X0_num_Sum_Encoded_After = 0

for i in range(len(XY0)//2):
    X0_num_Encoded_After = XY0_Encoded_After[-t]*2**z
    X0_num_Sum_Encoded_After += X0_num_Encoded_After
    t = t+1
    z = z+1
    
p = 0
u = 1+(len(XY0)//2)
Y0_num_Sum_Encoded_After = 0

for i in range(len(XY0)//2):
    Y0_num_Encoded_After = XY0_Encoded_After[-u]*2**p
    Y0_num_Sum_Encoded_After += Y0_num_Encoded_After
    u = u+1
    p = p+1


Decoded_X_Encoded_After = (X0_num_Sum_Encoded_After*Precision_X)+a_X
Decoded_Y_Encoded_After = (Y0_num_Sum_Encoded_After*Precision_Y)+a_Y

print()
print("Decoded_X_Encoded_After:",Decoded_X_Encoded_After)
print("Decoded_Y_Encoded_After:",Decoded_Y_Encoded_After)



