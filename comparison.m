
[labels, X] = libsvmread('scaled_train_data');
[labels_test, X_test] = libsvmread('scaled_test_data');

 
[classestimate,model]=Logistic_adaboost('train',X,labels,1900);

%%% T = 1900

% Classify the testdata with the trained model
testclass=Logistic_adaboost('apply',X_test,model);

acc_l = sum(testclass==labels_test) ./ size(testclass,1);
% end

[classestimate,model]=adaboost('train',X,labels,1900);

% Classify the testdata with the trained model
testclass=adaboost('apply',X_test,model);

acc = sum(testclass==labels_test) ./ size(testclass,1);
% end


hFig = figure;
algo = {'Logistic AdaBoost';'AdaBoost'};
density = [acc_l acc];
bar(density);
grid on;
xticklabels(algo);


%%% T = 300

 
[classestimate,model]=Logistic_adaboost('train',X,labels,300);

% Classify the testdata with the trained model
testclass=Logistic_adaboost('apply',X_test,model);

acc_l = sum(testclass==labels_test) ./ size(testclass,1);
% end

[classestimate,model]=adaboost('train',X,labels,300);

% Classify the testdata with the trained model
testclass=adaboost('apply',X_test,model);

acc = sum(testclass==labels_test) ./ size(testclass,1);
% end

hFig = figure;
algo = {'Logistic AdaBoost';'AdaBoost'};
density = [acc_l acc];
bar(density);
grid on;
xticklabels(algo);

