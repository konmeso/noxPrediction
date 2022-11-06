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

pyplot()


data=open("Data2021.csv")

plot_array_1d=Any[]

plot_array_2d=Any[]

col_names= ["Time", "NOx", "Cons", "lamda", "EGMF", "IntPr", "TR", "r", "EnTor","EGR","ExGasTemp"]

for i=3:length(col_names)
    plt1d= histogram(df[!,i] , bins=100 , alpha=1 ,
        xlabel=col_names[i] , ylabel="Frequency")
    plt2d= histogram2d(df[!,i] , df[!,9], bins=100, alpha=1 ,
        title=[col_names[i] ,"EnTor"])
    push!(plot_array_1d,plt1d)
    push!(plot_array_2d,plt2d)
end


#Plots.scatter(df[10000:20000,1],df[10000:20000,2],size=(500,500))
#d1=Plots.plot([p for p in plot_array_1d]... ,size=(1000,1000))
d2=Plots.plot([t for t in plot_array_2d]... , size=(1000,1000),layout=(3,3))
