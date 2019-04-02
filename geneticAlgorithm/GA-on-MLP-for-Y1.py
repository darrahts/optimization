import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random as rd
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

# Loading the data, shuffling and preprocessing it

Data = pd.read_csv("C:/Users/Dana Bani-Hani/Desktop/My Files/My Courses/Machine Learning Optimization Using Genetic Algorithm/Dataset.csv")
Data.sample(frac=1)
X1 = pd.DataFrame(Data, columns = ['X1','X2','X3','X4','X5','X6','X7','X8'])


Y1 = pd.DataFrame(Data, columns = ['Y1']).values
Y2 = pd.DataFrame(Data, columns = ['Y2']).values

Xbef = pd.get_dummies(X1,columns = ['X6','X8']).values

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(Xbef)

Y = Y1[:,0]

Cnt1 = len(X)
      
# 10, 4, 0, 1, 0, 1, 1, 0, 0

### The solver has no crossover because mutation is enough, since it only has two values
### VARIABLES ###
### VARIABLES ###
p_c_con = 1 # Probability of crossover
p_c_comb = 0.3 # Probability of crossover for integers
p_m_con = 0.1 # Probability of mutation
p_m_comb = 0.2 # Probability of mutation for integers
p_m_solver = 0.3 # Probability of mutation for the solver
K = 3 # For Tournament selection
pop = 50 # Population per generation
gen = 30 # Number of generations
ii2 = 3 # Number of K
### VARIABLES ###
### VARIABLES ###


### Combinatorial ###
UB_X1 = 10 # X1, Number of Neurons
LB_X1 = 6
UB_X2 = 8 # X2, Number of Hidden Layers
LB_X2 = 3



### Continuous ###
# Where the first 15 represent X3 and the second 15 represent X4
XY0 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
                1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]) # Initial solution

Init_Sol = XY0.copy()

n_list = np.empty((0,len(XY0)+2))
n_list_ST = np.empty((0,len(XY0)+2))
Sol_Here = np.empty((0,len(XY0)+2))
Sol_Here_ST = np.empty((0,1))

Solver_Type = ['adam']

for i in range(pop): # Shuffles the elements in the vector n times and stores them
    ST = rd.choice(Solver_Type)
    X1 = rd.randrange(6,10,2)
    X2 = rd.randrange(3,8,1)
    rd.shuffle(XY0)
    Sol_Here = np.append((X1,X2),XY0)
    n_list_ST = np.append(n_list_ST,ST)
    n_list = np.vstack((n_list,Sol_Here))


# Calculating fitness value

# X3 = Learning Rate
a_X3 = 0.01 # Lower bound of X
b_X3 = 0.3 # Upper bound of X
l_X3 = (len(XY0)/2) # Length of Chrom. X

# X4 = Momentum
a_X4 = 0.01 # Lower bound of Y
b_X4 = 0.99 # Upper bound of Y
l_X4 = (len(XY0)/2) # Length of Chrom. Y


Precision_X = (b_X3 - a_X3)/((2**l_X3)-1)

Precision_Y = (b_X4 - a_X4)/((2**l_X4)-1)


z = 0
t = 1
X0_num_Sum = 0

for i in range(len(XY0)//2):
    X0_num = XY0[-t]*(2**z)
    X0_num_Sum += X0_num
    t = t+1
    z = z+1


p = 0
u = 1 + (len(XY0)//2)
Y0_num_Sum = 0

for j in range(len(XY0)//2):
    Y0_num = XY0[-u]*(2**p)
    Y0_num_Sum += Y0_num
    u = u+1
    p = p+1


Decoded_X3 = (X0_num_Sum * Precision_X) + a_X3
Decoded_X4 = (Y0_num_Sum * Precision_Y) + a_X4


print()
print("Decoded_X3:",Decoded_X3)
print("Decoded_X4:",Decoded_X4)


For_Plotting_the_Best = np.empty((0,len(Sol_Here)+1))

One_Final_Guy = np.empty((0,len(Sol_Here)+2))
One_Final_Guy_Final = []

Min_for_all_Generations_for_Mut_1 = np.empty((0,len(Sol_Here)+1))
Min_for_all_Generations_for_Mut_2 = np.empty((0,len(Sol_Here)+1))

Min_for_all_Generations_for_Mut_1_1 = np.empty((0,len(Sol_Here)+2))
Min_for_all_Generations_for_Mut_2_2 = np.empty((0,len(Sol_Here)+2))

Min_for_all_Generations_for_Mut_1_1_1 = np.empty((0,len(Sol_Here)+2))
Min_for_all_Generations_for_Mut_2_2_2 = np.empty((0,len(Sol_Here)+2))


Generation = 1 


for i in range(gen):
    
    
    New_Population = np.empty((0,len(Sol_Here))) # Saving the new generation
    
    All_in_Generation_X_1 = np.empty((0,len(Sol_Here)+1))
    All_in_Generation_X_2 = np.empty((0,len(Sol_Here)+1))
    
    Min_in_Generation_X_1 = []
    Min_in_Generation_X_2 = []
    
    
    Save_Best_in_Generation_X = np.empty((0,len(Sol_Here)+1))
    Final_Best_in_Generation_X = []
    Worst_Best_in_Generation_X = []
    
    
    print()
    print("--> GENERATION: #",Generation)
    
    Family = 1

    for j in range(int(pop/2)): # range(int(pop/2))
            
        print()
        print("--> FAMILY: #",Family)
              
            
        # Tournament Selection
        # Tournament Selection
        # Tournament Selection
        
        Parents = np.empty((0,len(Sol_Here)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Warrior_1_index = np.random.randint(0,len(n_list)) #3
            Warrior_2_index = np.random.randint(0,len(n_list)) #5
            Warrior_3_index = np.random.randint(0,len(n_list))
            
            while Warrior_1_index == Warrior_2_index:
                Warrior_1_index = np.random.randint(0,len(n_list))
            while Warrior_2_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            while Warrior_1_index == Warrior_3_index:
                    Warrior_3_index = np.random.randint(0,len(n_list))
            
            Warrior_1 = n_list[Warrior_1_index,:]
            Warrior_2 = n_list[Warrior_2_index,:]
            Warrior_3 = n_list[Warrior_3_index,:]
            
            
            Battle_Troops = [Warrior_1,Warrior_2,Warrior_3]
            
            
            # For Warrior #1
            
            W1_Comb_1 = Warrior_1[0]
            W1_Comb_1 = int(W1_Comb_1)
            W1_Comb_2 = Warrior_1[1]
            W1_Comb_2 = int(W1_Comb_2)
            
            W1_Con = Warrior_1[2:]
            
            X0_num_Sum_W1 = 0
            Y0_num_Sum_W1 = 0
            
            z = 0
            t = 1
            OF_So_Far_W1 = 0
            
            for i in range(len(XY0)//2):
                X0_num_W1 = W1_Con[-t]*(2**z)
                X0_num_Sum_W1 += X0_num_W1
                t = t+1
                z = z+1
                
            p = 0
            u = 1 + (len(XY0)//2)
            
            for j in range(len(XY0)//2):
                Y0_num_W1 = W1_Con[-u]*(2**p)
                Y0_num_Sum_W1 += Y0_num_W1
                u = u+1
                p = p+1
                
        
            Decoded_X3_W1 = (X0_num_Sum_W1 * Precision_X) + a_X3
            Decoded_X4_W1 = (Y0_num_Sum_W1 * Precision_Y) + a_X4
            '''
            print()
            print("X0_num_W1:",X0_num_W1)
            print("Y0_num_W1:",Y0_num_W1)
            print("X0_num_Sum_W1:",X0_num_Sum_W1)
            print("Y0_num_Sum_W1:",Y0_num_Sum_W1)
            
            print("Decoded_X_W1:", Decoded_X_W1)
            print("Decoded_Y_W1:",Decoded_Y_W1)
            '''  
            
            Emp_3 = 0

            kf = cross_validation.KFold(Cnt1, n_folds=ii2)
            
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                Hid_Lay = ()
    
                # Objective Function
                
                #8,8,8
                
                for i in range(W1_Comb_2):
                    Hid_Lay = Hid_Lay + (W1_Comb_1,)
                  
                model1 = MLPRegressor(activation='relu',hidden_layer_sizes=Hid_Lay,
                                       learning_rate_init=Decoded_X3_W1,momentum=Decoded_X4_W1)
        
                model1.fit(X_train, Y_train)
                PL1=model1.predict(X_test)
                AC1=model1.score(X_test,Y_test)
            
                OF_So_Far_3 = 1-(model1.score(X_test,Y_test))
                
                
                Emp_3 += OF_So_Far_3
            
            OF_So_Far_W1 = Emp_3/ii2
            
            '''
            print()
            print("OF_So_Far_W1:",(1-OF_So_Far_W1))
            '''
            Prize_Warrior_1 = OF_So_Far_W1
            
            
            # For Warrior #2
            
            W2_Comb_1 = Warrior_2[0]
            W2_Comb_1 = int(W2_Comb_1)
            W2_Comb_2 = Warrior_2[1]
            W2_Comb_2 = int(W2_Comb_2)
            
            W2_Con = Warrior_2[2:]
            
            X0_num_Sum_W2 = 0
            Y0_num_Sum_W2 = 0
            
            z = 0
            t = 1
            OF_So_Far_W2 = 0
        
            for i in range(len(XY0)//2):
                X0_num_W2 = W2_Con[-t]*(2**z)
                X0_num_Sum_W2 += X0_num_W2
                t = t+1
                z = z+1
                
            p = 0
            u = 1 + (len(XY0)//2)
            
            for j in range(len(XY0)//2):
                Y0_num_W2 = W2_Con[-u]*(2**p)
                Y0_num_Sum_W2 += Y0_num_W2
                u = u+1
                p = p+1
                
        
            Decoded_X3_W2 = (X0_num_Sum_W2 * Precision_X) + a_X3
            Decoded_X4_W2 = (Y0_num_Sum_W2 * Precision_Y) + a_X4
            '''
            print()
            print("X0_num_W2:",X0_num_W2)
            print("Y0_num_W2:",Y0_num_W2)
            print("X0_num_Sum_W2:",X0_num_Sum_W2)
            print("Y0_num_Sum_W2:",Y0_num_Sum_W2)
            
            print("Decoded_X_W2:", Decoded_X_W2)
            print("Decoded_Y_W2:",Decoded_Y_W2)
            '''  
            
            Emp_4 = 0

            kf = cross_validation.KFold(Cnt1, n_folds=ii2)
            
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                Hid_Lay = ()
    
                # Objective Function
                
                for i in range(W2_Comb_2):
                    Hid_Lay = Hid_Lay + (W2_Comb_1,)
                  
                model1 = MLPRegressor(activation='relu',hidden_layer_sizes=Hid_Lay,
                                       learning_rate_init=Decoded_X3_W2,momentum=Decoded_X4_W2)
        
                model1.fit(X_train, Y_train)
                PL1=model1.predict(X_test)
                AC1=model1.score(X_test,Y_test)
            
                OF_So_Far_4 = 1-(model1.score(X_test,Y_test))
                
                
                Emp_4 += OF_So_Far_4
            
            OF_So_Far_W2 = Emp_4/ii2
            
            '''
            print()
            print("OF_So_Far_W2:",(1-OF_So_Far_W2))
            '''
            Prize_Warrior_2 = OF_So_Far_W2
            
            
            # For Warrior #3
            
            W3_Comb_1 = Warrior_3[0]
            W3_Comb_1 = int(W3_Comb_1)
            W3_Comb_2 = Warrior_3[1]
            W3_Comb_2 = int(W3_Comb_2)
            
            W3_Con = Warrior_3[2:]
            
            X0_num_Sum_W3 = 0
            Y0_num_Sum_W3 = 0
            
            z = 0
            t = 1
            OF_So_Far_W3 = 0
        
            for i in range(len(XY0)//2):
                X0_num_W3 = W3_Con[-t]*(2**z)
                X0_num_Sum_W3 += X0_num_W3
                t = t+1
                z = z+1
                
            p = 0
            u = 1 + (len(XY0)//2)
            
            for j in range(len(XY0)//2):
                Y0_num_W3 = W3_Con[-u]*(2**p)
                Y0_num_Sum_W3 += Y0_num_W3
                u = u+1
                p = p+1
                
        
            Decoded_X3_W3 = (X0_num_Sum_W3 * Precision_X) + a_X3
            Decoded_X4_W3 = (Y0_num_Sum_W3 * Precision_Y) + a_X4
            '''
            print()
            print("X0_num_W3:",X0_num_W3)
            print("Y0_num_W3:",Y0_num_W3)
            print("X0_num_Sum_W3:",X0_num_Sum_W3)
            print("Y0_num_Sum_W3:",Y0_num_Sum_W3)
            
            print("Decoded_X_W3:", Decoded_X_W3)
            print("Decoded_Y_W3:",Decoded_Y_W3)
            '''  
            
            Emp_5 = 0

            kf = cross_validation.KFold(Cnt1, n_folds=ii2)
            
            for train_index, test_index in kf:
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                Hid_Lay = ()
    
                # Objective Function
                
                for i in range(W3_Comb_2):
                    Hid_Lay = Hid_Lay + (W3_Comb_1,)
                  
                model1 = MLPRegressor(activation='relu',hidden_layer_sizes=Hid_Lay,
                                       learning_rate_init=Decoded_X3_W3,momentum=Decoded_X4_W3)
        
                model1.fit(X_train, Y_train)
                PL1=model1.predict(X_test)
                AC1=model1.score(X_test,Y_test)
            
                OF_So_Far_5 = 1-(model1.score(X_test,Y_test))
                
                
                Emp_5 += OF_So_Far_5
            
            OF_So_Far_W3 = Emp_5/ii2
            
            '''
            print()
            print("OF_So_Far_W3:",(1-OF_So_Far_W3))
            '''
            Prize_Warrior_3 = OF_So_Far_W3 
            
            '''
            print()
            print("OF_So_Far_W3:",(1-OF_So_Far_W3))
            '''
            Prize_Warrior_3 = OF_So_Far_W3
            
            
            if Prize_Warrior_1 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_1
                Winner_str = "Warrior_1"
                Prize = Prize_Warrior_1
            elif Prize_Warrior_2 == min(Prize_Warrior_1,Prize_Warrior_2,Prize_Warrior_3):
                Winner = Warrior_2
                Winner_str = "Warrior_2"
                Prize = Prize_Warrior_2
            else:
                Winner = Warrior_3
                Winner_str = "Warrior_3"
                Prize = Prize_Warrior_3
            '''
            print()
            print("Prize_Warrior_1:",Prize_Warrior_1)
            print("Prize_Warrior_2:",Prize_Warrior_2)
            print("Prize_Warrior_3:",Prize_Warrior_3)
            print("Winner is:",Winner_str,"at:",Prize)
            '''
            Parents = np.vstack((Parents,Winner))
        '''
        print()
        print("Parents:",Parents)
        '''
        
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        # Crossover
        # Crossover
        # Crossover
        
        
        Child_1 = np.empty((0,len(Sol_Here)))
        Child_2 = np.empty((0,len(Sol_Here)))
        
        
        # Crossover the Integers
        # Combinatorial
        
        # For X1
        # For X1
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X1
            Int_X1_1 = Parent_2[0]
            Int_X1_2 = Parent_1[0]
        else:
            # For X1
            Int_X1_1 = Parent_1[0]
            Int_X1_2 = Parent_2[0]
        
        # For X2
        # For X2
        Ran_CO_1 = np.random.rand()
        if Ran_CO_1 < p_c_comb:
            # For X2
            Int_X2_1 = Parent_2[1]
            Int_X2_2 = Parent_1[1]
        else:
            # For X2
            Int_X2_1 = Parent_1[1]
            Int_X2_2 = Parent_2[1]
        
        
        # Continuous
        # Where to crossover
        # Two-point crossover
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c_con:
        
            Cr_1 = np.random.randint(2,len(Sol_Here))
            Cr_2 = np.random.randint(2,len(Sol_Here))
                
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(2,len(Sol_Here))
            
            if Cr_1 < Cr_2:
                
                Cr_2 = Cr_2 + 1
                
                Copy_1 = Parent_1[2:]
                Mid_Seg_1 = Parent_1[Cr_1:Cr_2]
                
                Copy_2 = Parent_2[2:]
                Mid_Seg_2 = Parent_2[Cr_1:Cr_2]
                
                First_Seg_1 = Parent_1[2:Cr_1]
                Second_Seg_1 = Parent_1[Cr_2:]
                
                First_Seg_2 = Parent_2[2:Cr_1]
                Second_Seg_2 = Parent_2[Cr_2:]
                
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
                
                Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1))###
                Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2))
            else:
                
                Cr_1 = Cr_1 + 1
                
                Copy_1 = Parent_1[2:]
                Mid_Seg_1 = Parent_1[Cr_2:Cr_1]
                
                Copy_2 = Parent_2[2:]
                Mid_Seg_2 = Parent_2[Cr_2:Cr_1]
                
                First_Seg_1 = Parent_1[2:Cr_2]
                Second_Seg_1 = Parent_1[Cr_1:]
                
                First_Seg_2 = Parent_2[2:Cr_2]
                Second_Seg_2 = Parent_2[Cr_1:]
                
                Child_1 = np.concatenate((First_Seg_1,Mid_Seg_2,Second_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Mid_Seg_1,Second_Seg_2))
                Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1))###
                Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2))
        else:
            
            Child_1 = Parent_1[2:]
            Child_2 = Parent_2[2:]
            '''
            print()
            print("Child #1 here2:",Child_1)
            print("Child #2 here2:",Child_2)
            '''
            Child_1 = np.insert(Child_1,0,(Int_X1_1,Int_X2_1))###
            Child_2 = np.insert(Child_2,0,(Int_X1_2,Int_X2_2))
            
            
        '''    
        print()
        print("Child #1:",Child_1)
        print("Child #2:",Child_2)
        '''
        '''
        print("Cr_1:",Cr_1)
        print("Cr_2:",Cr_2)
        print("Parent #1:",Parent_1)
        print("Parent #2:",Parent_2)
        print("Child #1:",Child_1)
        print("Child #2:",Child_2)
        '''
        
        
        # Mutation Child #1
        # Mutation Child #1
        # Mutation Child #1
        
        Mutated_Child_1 = []
        
        
        # Combinatorial
        
        # For X1
        # For X1
        Ran_M1_1 = np.random.rand()
        if Ran_M1_1 < p_m_comb:
            Ran_M1_2 = np.random.rand()
            if Ran_M1_2 >= 0.5:
                if Child_1[0] == UB_X1:
                    C_X1_M1 = Child_1[0]
                elif Child_1[0] == LB_X1:
                    C_X1_M1 = Child_1[0]
                else:
                    C_X1_M1 = Child_1[0] + 2
            else:
                if Child_1[0] == UB_X1:
                    C_X1_M1 = Child_1[0]
                elif Child_1[0] == LB_X1:
                    C_X1_M1 = Child_1[0]
                else:
                    C_X1_M1 = Child_1[0] - 2
        else:
            C_X1_M1 = Child_1[0]
        
        # For X2
        # For X2
        Ran_M1_3 = np.random.rand()
        if Ran_M1_3 < p_m_comb:
            Ran_M1_4 = np.random.rand()
            if Ran_M1_4 >= 0.5:
                if Child_1[1] == UB_X2:
                    C_X2_M1 = Child_1[1]
                elif Child_1[1] == LB_X2:
                    C_X2_M1 = Child_1[1]
                else:
                    C_X2_M1 = Child_1[1] + 1
            else:
                if Child_1[1] == UB_X2:
                    C_X2_M1 = Child_1[1]
                elif Child_1[1] == LB_X2:
                    C_X2_M1 = Child_1[1]
                else:
                    C_X2_M1 = Child_1[1] - 1
        else:
            C_X2_M1 = Child_1[1]
           
        
        # Continuous
        
        t = 0
        
        Child_1n = Child_1[2:]
        
        for i in Child_1n:
        
            Ran_Mut_1 = np.random.rand() # Probablity to Mutate
            
            if Ran_Mut_1 < p_m_con: # If probablity to mutate is less than p_m, then mutate
                
                if Child_1n[t] == 0:
                    Child_1n[t] = 1
                else:
                    Child_1n[t] = 0
                t = t+1
            
                Mutated_Child_1n = Child_1n
                
            else:
                Mutated_Child_1n = Child_1n
        
        Mutated_Child_1 = np.insert(Mutated_Child_1n,0,(C_X1_M1,C_X2_M1))
        
        '''
        print()
        print("Mutated_Child #1:",Mutated_Child_1)
        '''
        
        # Mutation Child #2
        # Mutation Child #2
        # Mutation Child #2
        
        Mutated_Child_2 = []
        
        
        # Combinatorial
        
        # For X1
        # For X1
        Ran_M2_1 = np.random.rand()
        if Ran_M2_1 < p_m_comb:
            Ran_M2_2 = np.random.rand()
            if Ran_M2_2 >= 0.5:
                if Child_2[0] == UB_X1:
                    C_X1_M2 = Child_1[0]
                elif Child_2[0] == LB_X1:
                    C_X1_M2 = Child_2[0]
                else:
                    C_X1_M2 = Child_2[0] + 2
            else:
                if Child_2[0] == UB_X1:
                    C_X1_M2 = Child_2[0]
                elif Child_1[0] == LB_X1:
                    C_X1_M2 = Child_2[0]
                else:
                    C_X1_M2 = Child_1[0] - 2
        else:
            C_X1_M2 = Child_2[0]
        
        # For X2
        # For X2
        Ran_M2_3 = np.random.rand()
        if Ran_M2_3 < p_m_comb:
            Ran_M2_4 = np.random.rand()
            if Ran_M2_4 >= 0.5:
                if Child_2[1] == UB_X2:
                    C_X2_M2 = Child_2[1]
                elif Child_2[1] == LB_X2:
                    C_X2_M2 = Child_2[1]
                else:
                    C_X2_M2 = Child_2[1] + 1
            else:
                if Child_2[1] == UB_X2:
                    C_X2_M2 = Child_2[1]
                elif Child_2[1] == LB_X2:
                    C_X2_M2 = Child_2[1]
                else:
                    C_X2_M2 = Child_2[1] - 1
        else:
            C_X2_M2 = Child_2[1]
           
        
        # Continuous
        
        t = 0
        
        Child_2n = Child_2[2:]
        
        for i in Child_2n:
        
            Ran_Mut_2 = np.random.rand() # Probablity to Mutate
            
            if Ran_Mut_2 < p_m_con: # If probablity to mutate is less than p_m, then mutate
                
                if Child_2n[t] == 0:
                    Child_2n[t] = 1
                else:
                    Child_2n[t] = 0
                t = t+1
            
                Mutated_Child_2n = Child_2n
                
            else:
                Mutated_Child_2n = Child_2n
        
        Mutated_Child_2 = np.insert(Mutated_Child_2n,0,(C_X1_M2,C_X2_M2))
        
        '''
        print()
        print("Mutated_Child #2:",Mutated_Child_2)
        '''
        
        # Calculate fitness values of mutated children
        
        fit_val_muted_children = np.empty((0,2))
        
        
        # For mutated child #1
        
        MC_1_Comb_1 = Mutated_Child_1[0]
        MC_1_Comb_1 = int(MC_1_Comb_1)
        MC_1_Comb_2 = Mutated_Child_1[1]
        MC_1_Comb_2 = int(MC_1_Comb_2)
        
        MC_1_Con = Mutated_Child_1[2:]
        
        X0_num_Sum_MC_1 = 0
        Y0_num_Sum_MC_1 = 0
        
        z = 0
        t = 1
        OF_So_Far_MC_1 = 0
    
        for i in range(len(XY0)//2):
            X0_num_MC_1 = MC_1_Con[-t]*(2**z)
            X0_num_Sum_MC_1 += X0_num_MC_1
            t = t+1
            z = z+1
            
        p = 0
        u = 1 + (len(XY0)//2)
        
        for j in range(len(XY0)//2):
            Y0_num_MC_1 = MC_1_Con[-u]*(2**p)
            Y0_num_Sum_MC_1 += Y0_num_MC_1
            u = u+1
            p = p+1
            
        Decoded_X3_MC_1 = (X0_num_Sum_MC_1 * Precision_X) + a_X3
        Decoded_X4_MC_1 = (Y0_num_Sum_MC_1 * Precision_Y) + a_X4
        
        
        Emp_6 = 0

        kf = cross_validation.KFold(Cnt1, n_folds=ii2)
        
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            Hid_Lay = ()

            # Objective Function
            
            for i in range(MC_1_Comb_2):
                Hid_Lay = Hid_Lay + (MC_1_Comb_1,)
              
            model1 = MLPRegressor(activation='relu',hidden_layer_sizes=Hid_Lay,
                                   learning_rate_init=Decoded_X3_MC_1,momentum=Decoded_X4_MC_1)
            model1.fit(X_train, Y_train)
            PL1=model1.predict(X_test)
            AC1=model1.score(X_test,Y_test)
        
            OF_So_Far_6 = 1-(model1.score(X_test,Y_test))
            
            Emp_6 += OF_So_Far_6
        
        OF_So_Far_MC_1 = Emp_6/ii2
        
        
        # For mutated child #2
        
        MC_2_Comb_1 = Mutated_Child_2[0]
        MC_2_Comb_1 = int(MC_2_Comb_1)
        MC_2_Comb_2 = Mutated_Child_2[1]
        MC_2_Comb_2 = int(MC_2_Comb_2)
        
        MC_2_Con = Mutated_Child_2[2:]
        
        X0_num_Sum_MC_2 = 0
        Y0_num_Sum_MC_2 = 0
        
        z = 0
        t = 1
        OF_So_Far_MC_2 = 0
    
        for i in range(len(XY0)//2):
            X0_num_MC_2 = MC_2_Con[-t]*(2**z)
            X0_num_Sum_MC_2 += X0_num_MC_2
            t = t+1
            z = z+1
            
        p = 0
        u = 1 + (len(XY0)//2)
        
        for j in range(len(XY0)//2):
            Y0_num_MC_2 = MC_2_Con[-u]*(2**p)
            Y0_num_Sum_MC_2 += Y0_num_MC_2
            u = u+1
            p = p+1
            
        Decoded_X3_MC_2 = (X0_num_Sum_MC_2 * Precision_X) + a_X3
        Decoded_X4_MC_2 = (Y0_num_Sum_MC_2 * Precision_Y) + a_X4
        
        
        Emp_7 = 0

        kf = cross_validation.KFold(Cnt1, n_folds=ii2)
        
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            
            Hid_Lay = ()

            # Objective Function
            
            for i in range(MC_2_Comb_2):
                Hid_Lay = Hid_Lay + (MC_2_Comb_1,)
              
            model1 = MLPRegressor(activation='relu',hidden_layer_sizes=Hid_Lay,
                                   learning_rate_init=Decoded_X3_MC_2,momentum=Decoded_X4_MC_2)
            model1.fit(X_train, Y_train)
            PL1=model1.predict(X_test)
            AC1=model1.score(X_test,Y_test)
        
            OF_So_Far_7 = 1-(model1.score(X_test,Y_test))
            
            Emp_7 += OF_So_Far_7
        
        OF_So_Far_MC_2 = Emp_7/ii2
        
        '''
        print()
        print("FV at Mutated Child #1 at Gen #",Generation,":", OF_So_Far_MC_1)
        print("FV at Mutated Child #2 at Gen #",Generation,":", OF_So_Far_MC_2)
        '''   
              
        All_in_Generation_X_1_1_temp = Mutated_Child_1[np.newaxis]
        All_in_Generation_X_1_1 = np.column_stack((OF_So_Far_MC_1, All_in_Generation_X_1_1_temp))
        
        
        All_in_Generation_X_2_1_temp = Mutated_Child_2[np.newaxis]
        All_in_Generation_X_2_1 = np.column_stack((OF_So_Far_MC_2, All_in_Generation_X_2_1_temp))
        
        
        All_in_Generation_X_1 = np.vstack((All_in_Generation_X_1,All_in_Generation_X_1_1))
        All_in_Generation_X_2 = np.vstack((All_in_Generation_X_2,All_in_Generation_X_2_1))
        
        
        Save_Best_in_Generation_X = np.vstack((All_in_Generation_X_1,All_in_Generation_X_2))
        
        
        New_Population = np.vstack((New_Population,Mutated_Child_1,Mutated_Child_2))
        
        t = 0
        R_1 = []
        for i in All_in_Generation_X_1:
            
            if (All_in_Generation_X_1[t,:1]) <= min(All_in_Generation_X_1[:,:1]):
                R_1 = All_in_Generation_X_1[t,:]
            t = t+1
            
        
        Min_in_Generation_X_1 = R_1[np.newaxis]
        '''
        print()
        print("Min_in_Generation_X_1:",Min_in_Generation_X_1)
        '''
        t = 0
        R_2 = []
        for i in All_in_Generation_X_2:
            
            if (All_in_Generation_X_2[t,:1]) <= min(All_in_Generation_X_2[:,:1]):
                R_2 = All_in_Generation_X_2[t,:]
            t = t+1
                
        
        Min_in_Generation_X_2 = R_2[np.newaxis]
        '''
        print()
        print("Min_in_Generation_X_2:",Min_in_Generation_X_2)
        '''
        
        Family = Family+1
    
    '''
    print()
    print("New_Population Before:",New_Population)
    '''
    t = 0
    R_Final = []
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) <= min(Save_Best_in_Generation_X[:,:1]):
            R_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    
    Final_Best_in_Generation_X = R_Final[np.newaxis]
    '''
    print()
    print("Final_Best_in_Generation_X:",Final_Best_in_Generation_X)
    '''
    t = 0
    R_22_Final = []
    for i in Save_Best_in_Generation_X:
        
        if (Save_Best_in_Generation_X[t,:1]) >= max(Save_Best_in_Generation_X[:,:1]):
            R_22_Final = Save_Best_in_Generation_X[t,:]
        t = t+1
            
    
    Worst_Best_in_Generation_X = R_22_Final[np.newaxis]
    '''
    print()
    print("Worst_Best_in_Generation_X:",Worst_Best_in_Generation_X)
    '''
    
    # Elitism, the best in the generation lives
    # Elitism, the best in the generation lives
    # Elitism, the best in the generation lives
    
    Darwin_Guy = Final_Best_in_Generation_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Generation_X[:]
    
    
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    '''
    print()
    print("Darwin_Guy:",Darwin_Guy)
    print("Not_So_Darwin_Guy:",Not_So_Darwin_Guy)
    
    print()
    print("Before:",New_Population)
    print()
    '''
    Best_1 = np.where((New_Population == Darwin_Guy).all(axis=1))
    Worst_1 = np.where((New_Population == Not_So_Darwin_Guy).all(axis=1))
    '''
    print()
    print("Best_1:",Best_1)
    print("Worst_1:",Worst_1)
    '''
    New_Population[Worst_1] = Darwin_Guy
    '''
    print("New_Population After:",New_Population)
    
    print()
    print("After:",New_Population)
    '''
    n_list = New_Population
    
    '''
    print()
    print("The New Population Are:\n",New_Population)
    '''
    
    Min_for_all_Generations_for_Mut_1 = np.vstack((Min_for_all_Generations_for_Mut_1,Min_in_Generation_X_1))
    Min_for_all_Generations_for_Mut_2 = np.vstack((Min_for_all_Generations_for_Mut_2,Min_in_Generation_X_2))
    
    
    Min_for_all_Generations_for_Mut_1_1 = np.insert(Min_in_Generation_X_1, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2 = np.insert(Min_in_Generation_X_2, 0, Generation)
    '''
    Min_for_all_Generations_for_Mut_1_1_ST = np.insert(Min_in_Generation_X_1_ST, 0, Generation)
    Min_for_all_Generations_for_Mut_2_2_ST = np.insert(Min_in_Generation_X_2_ST, 0, Generation)
    '''
    Min_for_all_Generations_for_Mut_1_1_1 = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_1_1))
    Min_for_all_Generations_for_Mut_2_2_2 = np.vstack((Min_for_all_Generations_for_Mut_2_2_2,Min_for_all_Generations_for_Mut_2_2))
    
    
    Generation = Generation+1
    
    



One_Final_Guy = np.vstack((Min_for_all_Generations_for_Mut_1_1_1,Min_for_all_Generations_for_Mut_2_2_2))


t = 0
Final_Here = []
for i in One_Final_Guy:
    
    if (One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
        Final_Here = One_Final_Guy[t,:]
    t = t+1
        

One_Final_Guy_Final = Final_Here[np.newaxis]


XY0_Encoded_After = Final_Here[4:]

# DECODING
# DECODING
# DECODING

z = 0
t = 1
X0_num_Sum_Encoded_After = 0

for i in range(len(XY0)//2):
    X0_num_Encoded_After = XY0_Encoded_After[-t]*(2**z)
    X0_num_Sum_Encoded_After += X0_num_Encoded_After
    t = t+1
    z = z+1


p = 0
u = 1 + (len(XY0)//2)
Y0_num_Sum_Encoded_After = 0

for j in range(len(XY0)//2):
    Y0_num_Encoded_After = XY0_Encoded_After[-u]*(2**p)
    Y0_num_Sum_Encoded_After += Y0_num_Encoded_After
    u = u+1
    p = p+1


Decoded_X_After = (X0_num_Sum_Encoded_After * Precision_X) + a_X3
Decoded_Y_After = (Y0_num_Sum_Encoded_After * Precision_Y) + a_X4

print()
print()
print("The High Accuracy is:",(1-One_Final_Guy_Final[:,1]))
print("Number of Neurons:",Final_Here[2])
print("Number of Hidden Layers:",Final_Here[3])
print("Learning Rate:",Decoded_X_After)
print("Momentum:",Decoded_Y_After)




