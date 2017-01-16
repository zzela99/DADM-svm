%%
Mtrain = csvread('trainset.csv');
Mtest = csvread('testset.csv');
Mtest_features=Mtest(:,1:5);
Mtest_labels=Mtest(:,6);
Mtrain_features=Mtrain(:,1:5);
Mtrain_labels=Mtrain(:,6);

%% 

SVMStruct1 = svmtrain(Mtrain_features,Mtrain_labels,'kernel_function','quadratic');
wynik1 = svmclassify(SVMStruct1,Mtest_features)

sum=0;
s=size(Mtest_labels);
a=s(1);
i=0;
for i =1:a 
    
    if wynik1(i)== Mtest_labels(i)
        sum=sum + 1; 
        
    end
end
accuracyinpercents = sum/a*100 
