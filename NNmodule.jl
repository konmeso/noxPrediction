module NNcreation
    using Flux
    using Plots
    using DataFrames
    using StatsPlots
    using MLDataUtils
    using DelimitedFiles
    using ScikitLearn
    using PyCall
    using PyPlot
    using StatsBase
    using MultivariateStats
    @sk_import feature_selection: SelectKBest
    @sk_import feature_selection: f_regression
    @sk_import feature_selection: mutual_info_regression
    @sk_import decomposition: PCA #Principal Component Analysis


#------------------------------------------Feature Scaling Function--------------------------------------------------------
    function feature_scaling(X,y)

        Xf=StatsBase.fit(ZScoreTransform,X,dims=1)
        yf=StatsBase.fit(ZScoreTransform,y,dims=1)

        Xn=StatsBase.transform(Xf,X)
        yn=StatsBase.transform(yf,y)

        return Xn,yn,Xf,yf
    end

#---------------------Dimensionality Reduction/Feature Selection functions------------------------------------------
    function tests(Xn,yn,inputs::Array{String})

        ftest1=SelectKBest(mutual_info_regression,k="all")
        ftest2=SelectKBest(f_regression,k="all")

        ftest1.fit(Xn,yn)
        ftest2.fit(Xn,yn)

        pb1=Plots.bar(inputs,ftest1.scores_,label="Mutual Info Regression")
        pb2=Plots.bar(inputs,ftest2.scores_,label="F-Regression")

        Plots.plot(pb1,pb2,layout=(2,1))
    end

    function pca_calc(Xn) #PCA
        p_c_a=PCA(3).fit(Xn)
        Xpca=p_c_a.transform(Xn)
        Plots.plot(cumsum(p_c_a.explained_variance_ratio_), label ="")
    end

#-------------------------Custom Neural Network Function-----------------------------------------------
    using Flux: @epochs
    using Metrics
    using MLDataUtils


    function NeuralCreation(ins,outs,Loss::String="mse",Heta::Float64=0.1,Optimizer::String="Adam",Ne::Int64=5,knods::String="Known",Layer_knods::Array{Int64}=[length(ins'[:,1]),10,10,length(outs'[:,1])])


        shuffle_data=shuffleobs((ins',outs'))   # Shuffling the data to get an unbiased model

        (X_train , y_train) , (X_test , y_test) = splitobs(shuffle_data, at=0.8)    #Splitting the data to train and testing

        layers=Any[]    #Initialization
        numin=Any[]
        num=Any[]

        if knods=="Unknown"

            print("Write the number of layers you want")
            Nl=readline()
            Nl=parse(Int64,Nl)
            for t in 1:Nl  # For loop for layer creation using readline which demands user input
                if t==1
                    println("Input the number of inputs in the first hidden layer")
                    num=readline()
                    num=parse(Int64,num) #Converting string to int64
                    l=Dense(length(ins'[:,1]),num,relu)     #In the first layer the knots are equal to the inputs

                elseif t==Nl
                    l=Dense(numin[t-1],length(outs'[:,1]))   #In the last layer the knots are equal to the outputs

                else
                    println("Input the number of inputs in hidden layer" ,"\t" , t)       #Creating the hidden layers
                    num=readline()
                    num=parse(Int64,num)
                    l=Dense(numin[t-1],num,relu)
                end
                push!(numin,num)
                push!(layers,l)
            end
        elseif knods=="Known"
            for t in 1:(length(Layer_knods)-1)
                l=Dense(Layer_knods[t],Layer_knods[t+1],relu)
                if t==length(Layer_knods)-1
                    l=Dense(Layer_knods[t],Layer_knods[t+1])
                end
                push!(layers,l)
            end
        end

        model=Chain([t for t in layers ]...)  # Chaining the layers

        if Loss=="mse"              # Mean square error
            loss_fun(Xn,yn)=Flux.Losses.mse(model(Xn'),yn')    #Mean average error
        elseif Loss=="mae"
            loss_fun(Xn,yn)=Flux.Losses.mae(model(Xn'),yn')     # Hubber losses
        elseif Loss=="Huber"
            loss_fun(Xn,yn)=Flux.Losses.huber_loss(model(Xn'),yn')
        end


        if Optimizer=="Adam"        #Different types of optimizers
            opt= ADAM(Heta)
        elseif Optimizer=="Descent"
            opt=Descent(Heta)
        elseif Optimizer=="Momentum"
            println("Choose momentum ρ")
            ρ=readline()
            ρ=parse(Int64,ρ)
            opt=Momentum()
        elseif Optimizer=="Nesterov"
            println("Choose Nesterov momentum ρ")
            ρ=readline()
            ρ=parse(Int64,ρ)
            opt=Nesterov()
        end


        par=params(model)
        train = [(X_train',y_train')];

        @epochs Ne Flux.train!(loss_fun,par,train,opt)  #Training the model in the NN created


        y_pred=model(X_test)         #Predictions on the data

        test_acc=r2_score(y_test,y_pred) # Checking the accuracy with r2_metric (Best possible score 1)

        println("The R^2 score on the test set is ",(test_acc),"\n")

        Flux.reset!(model)

        return model,X_test,y_test,y_pred

    end

    export NeuralCreation,tests,feature_scaling,pca_calc,test_acc



end


#----------------------------------------------------------------------------------------------
