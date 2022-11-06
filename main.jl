using Flux
using Plots
using CSV
using DataFrames
using StatsPlots
using Noise
using MLDataUtils
using DelimitedFiles
using ScikitLearn
using PyCall
using PyPlot
using StatsBase
using MultivariateStats
using HDF5

gr()

f="Data2021.csv"

open(f,"r")
df=CSV.read(f, DataFrame  ; header=1:2 ,delim="\t" , ignorerepeated=true)


data=convert(Matrix,df)

include("NNmodule.jl")

using .NNcreation

col_names= ["Time", "NOx", "Cons", "lamda", "EGMF", "IntPr", "TR", "r", "EnTor","EGR","ExGasTemp"]



#-----------------------------------------NOx prediction model---------------------------------------------------
X_NOx=data[:,setdiff(1:end,(1,2))]
y_NOx=data[:,2]

inputs_NOx=[col_names[3],col_names[4],col_names[5],col_names[6],col_names[7],
    col_names[8],col_names[9],col_names[10],col_names[11]]

Xn_NOx,yn_NOx,Xf_NOx,yf_NOx=NNcreation.feature_scaling(X_NOx,y_NOx)
NNcreation.tests(Xn_NOx,yn_NOx,inputs_NOx)

NNcreation.pca_calc(Xn_NOx)

#ins=Xn_NOx[:,setdiff(1:end,(1,3,4,5,6,9))]
ins = Xn_NOx[:,setdiff(1:end,(4,5,9))]
outs=yn_NOx
Loss="mse"
Heta=0.01
Optimizer="Adam"
Ne=500

@time begin

    model_NOX,X_test_NOx,y_test_NOx,y_pred_NOx = NNcreation.NeuralCreation(ins,outs,Loss,Heta,Optimizer,Ne)

end


y_predicted=shuffleobs(StatsBase.reconstruct(yf_NOx,y_pred_NOx')) #Reconstruct: fitted -> actual values
y_tested=shuffleobs(StatsBase.reconstruct(yf_NOx,y_test_NOx'))

Plots.scatter(y_tested,y_predicted,label="Test-Predicted")





#-----------------------------------------Torque prediction model---------------------------------------------------

X_torque=data[:,setdiff(1:end,(1,7,9)) ]
y_torque=data[:,9]

inputs_torque=[col_names[2],col_names[3],col_names[4],col_names[5],
    col_names[6],col_names[8],col_names[10],col_names[11]]

Xn_torque,yn_torque,Xf_torque,yf_torque=NNcreation.feature_scaling(X_torque,y_torque)
NNcreation.tests(Xn_torque,yn_torque,inputs_torque)
PCA_torque=NNcreation.NNcreation.pca_calc(Xn_torque)

#ins=Xn_torque[:,setdiff(1:end,(3,5,6,7,8))]
ins=Xn_torque[:,setdiff(1:end,(5,7,8))]
outs=yn_torque
Loss="mse"
Heta=0.01
Optimizer="Adam"
Ne=500

@time begin
    model_torque,X_test_torque,y_test_torque,y_pred_torque = NNcreation.NeuralCreation(ins,outs,Loss,Heta,Optimizer,Ne)
end

y_predicted=shuffleobs(StatsBase.reconstruct(yf_torque,y_pred_torque'))
y_tested=shuffleobs(StatsBase.reconstruct(yf_torque,y_test_torque'))

Plots.scatter(y_tested,y_predicted,label="Test-Predicted")

#NNcreation.pca_calc(Xn_torque)

#---------------------------------lambda prediction model-----------------------------------------------------------
X_lambda=data[:,setdiff(1:end,(1,4)) ]
y_lambda=data[:,4]

inputs_lambda=[col_names[2],col_names[3],col_names[5],
    col_names[6],col_names[7],col_names[8],col_names[9],col_names[10],col_names[11]]

Xn_lambda,yn_lambda,Xf_lambda,yf_lambda=NNcreation.feature_scaling(X_lambda,y_lambda)
NNcreation.tests(Xn_lambda,yn_lambda,inputs_lambda)
PCA_lambda=NNcreation.NNcreation.pca_calc(Xn_lambda)

#ins=Xn_lambda[:,setdiff(1:end,(2,4,5,6,8,9))]
ins=Xn_lambda[:,setdiff(1:end,(2,4,5,6,8))]
#exclude Time,EGR,ExGasTemp
outs=yn_lambda
Loss="mse"
Heta=0.01
Optimizer="Adam"
Ne=500

@time begin
    model_lambda,X_test_lambda,y_test_lambda,y_pred_lambda = NNcreation.NeuralCreation(ins,outs,Loss,Heta,Optimizer,Ne)
end

y_predicted=shuffleobs(StatsBase.reconstruct(yf_lambda,y_pred_lambda'))
y_tested=shuffleobs(StatsBase.reconstruct(yf_lambda,y_test_lambda'))

Plots.scatter(y_tested,y_predicted,label="Test-Predicted")

#NNcreation.pca_calc(Xn_lambda)

#------------------------------Intake Pressure Prediction Model-------------------------------------------
X_IntPr=data[:,setdiff(1:end,(1,6)) ]
y_IntPr=data[:,6]

inputs_IntPr=[col_names[2],col_names[3],col_names[4],
    col_names[5],col_names[7],col_names[8],col_names[9],col_names[10],col_names[11]]

Xn_IntPr,yn_IntPr,Xf_IntPr,yf_IntPr=NNcreation.feature_scaling(X_IntPr,y_IntPr)
NNcreation.tests(Xn_IntPr,yn_IntPr,inputs_IntPr)
PCA_IntPr=NNcreation.NNcreation.pca_calc(Xn_lambda)

#ins=Xn_IntPr[:,setdiff(1:end,(1,3,5,6,7,8))]
ins=Xn_IntPr[:,setdiff(1:end,(5,6,7,8))]
#exclude Time,EGR,ExGasTemp
outs=yn_IntPr
Loss="mse"
Heta=0.01
Optimizer="Adam"
Ne=500

@time begin
    model_IntPr,X_test_IntPr,y_test_IntPr,y_pred_IntPr = NNcreation.NeuralCreation(ins,outs,Loss,Heta,Optimizer,Ne)
end

y_predicted=shuffleobs(StatsBase.reconstruct(yf_IntPr,y_pred_IntPr'))
y_tested=shuffleobs(StatsBase.reconstruct(yf_IntPr,y_test_IntPr'))

Plots.scatter(y_tested,y_predicted,label="Test-Predicted")

#NNcreation.pca_calc(Xn_IntPr)

#---------------------------------------Consumption prediction model----------------------------------------------
X_Cons=data[:,setdiff(1:end,(1,3)) ]
y_Cons=data[:,3]

inputs_Cons=[col_names[2],col_names[4],
    col_names[5],col_names[6],col_names[7],col_names[8],col_names[9],col_names[10],col_names[11]]

Xn_Cons,yn_Cons,Xf_Cons,yf_Cons=NNcreation.feature_scaling(X_Cons,y_Cons)
NNcreation.tests(Xn_Cons,yn_Cons,inputs_Cons)
#PCA_lambda=NNcreation.NNcreation.pca_calc(Xn_lambda)

#ins=Xn_Cons[:,setdiff(1:end,(1,2,5,6,8,9))]
ins=Xn_Cons[:,setdiff(1:end,(1,5,6,8))]
#exclude Time,EGR,ExGasTemp
outs=yn_Cons
Loss="mse"
Heta=0.01
Optimizer="Adam"
Ne=500

@time begin
    model_Cons,X_test_Cons,y_test_Cons,y_pred_Cons = NNcreation.NeuralCreation(ins,outs,Loss,Heta,Optimizer,Ne)
end

y_predicted=shuffleobs(StatsBase.reconstruct(yf_Cons,y_pred_Cons'))
y_tested=shuffleobs(StatsBase.reconstruct(yf_Cons,y_test_Cons'))

Plots.scatter(y_tested,y_predicted,label="Test-Predicted")

NNcreation.pca_calc(Xn_Cons)
