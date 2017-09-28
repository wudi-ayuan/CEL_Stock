

[features, target]=GenFeature();
%[shp, macd, vlt, dma50, rsi, vlm_dlt, tbill_dl, rtn_past, 'Open','High,'Low',Close','Volume','Adj Close' ];

% % %//////Try cancer dataset
% %  Data = csvread('Medical_Dataset.csv');
% %  features = Data(:,1:30);
% %  target = Data(:,31);
% % %/////
% ordering=randperm(size(features,1)); %Randomizing the dataset 
% features=features(ordering,:);
% target=target(ordering,:);

target=flipud(target);
features=flipud(features);
target=target+1;
csvwrite('sp500features',features);
csvwrite('target',target);
strategy=features(:,end-4:end);
features=features(:,1:end-5);
instnum=size(features,1);
train_ind=[1:instnum/10*8];
xtrain=features(train_ind,:);
ytrain=target(train_ind,:);
a=[1:instnum];
test_ind=setdiff(a,train_ind);
xtest=features(test_ind,:);
ytest=target(test_ind,:);





%%
%prediction: 1/trending up 0/tredning down
%%base classifier machine learning STofArt

X=xtrain;
Y=ytrain;
ml_Enom=4;
fn_Enom=5;
nAgents=ml_Enom;
strat_arg=false;
%bs learner: logistic,  SVM,        RandomForest,    LASSO
%fn learner: rsi,       vlm_dlt,    dma50,           blgbnd,   mnx
Feat_nam=['Sharpe Ratio    '; 'MACD            '; 'volatility      '; 'T-Bill Return   '; 'yesterday return'; 'OPen            ';    'High            ';    'Low             ';    'Close           ';    'Volume          ' ;   'Adj Close       '];
Feat_nam = cellstr(Feat_nam);
ML_nam{1}='logistic';  ML_nam{2}='SVM';ML_nam{3}='RandomForest';ML_nam{4}='LASSO';
FN_nam{1}='rsi';FN_nam{2}='vlm_dlt';FN_nam{3}='dma50';FN_nam{4}='blgbnd';FN_nam{5}='mnx';
k1=1;
k2=2;
k3=3;

%% Local Learner output:
% target_cat=categorical(target);
% for i=4:size(features,1)-1,
%         %Logistic / probability
%     logB=glmfit(features(1:i,:),target_cat(1:i),'binomial','link','logit');
%     score(i+1,1)=glmval(logB,features(i+1,:),'logit');
%         %SVM / probability
%     svmB=fitcsvm(features(1:i,:),target_cat(1:i),'KernelFunction','RBF','standardize', true);
%     svmB = fitPosterior(svmB);
%     [~,score2]=predict(svmB,features(i+1,:));
%     score(i+1,2)=score2(2);
%     i
%         %RBF / regression
%     rfrB = TreeBagger(10,features(1:i,:),target(1:i),'Method','regression'); 
%     score(i+1,3)=predict(rfrB, features(i+1,:));
%         %LASSO / regression
%     [lasB, info] = lasso(features(1:i,:),target(1:i),'NumLambda',40);
%     yhatlas=info.Intercept(5)+features(i+1,:)*lasB(:,5);
%     score(i+1,4)=yhatlas;
% 
% 
% end
% LLpred=[round(score(:,1:2))+1 (score(:,3:4)>1.5)+1];
load('LLprediction.mat')
kau=0.75;
%% CEL fn=true/false, ml_no=4, ml_fn=5;
[ Result , w, F_EL,P, predict_output,partition, CELscore] = Contextual_EL_WM(features, target, train_ind, test_ind, nAgents, strategy,strat_arg,LLpred,kau);

%% IBCC
bccSettings=settings.BccSettings;
predict_output=LLpred';
predict_output(:,1:4)=repmat(target(1:4)',nAgents,1);        %fill out blank
nScores=2;
weight=zeros(size(features,1),nAgents);
%performance evaluation
Count = zeros(5,1);     % For ensemble learner
Counting = zeros(5,nAgents);   % For average, best and worst learner

for t=test_ind,
inputC{1}=reshape(repmat([1:nAgents]',1,t),1,[]);                     %classfier IDs
inputC{2}=reshape(repmat(1:t,nAgents,1),1,[]);  %objects' IDs
inputC{3}=reshape(predict_output(:,1:t),1,[]); 
IBCC_target = zeros(1,t);
IBCC_target(1:t-1) =target(1:t-1)';
vbcmb=combiners.bcc.IbccVb(bccSettings, nAgents, IBCC_target, 2, 2);
[post_T1, sd_post_T1, post_Alpha1]=vbcmb.combineDecisions(inputC);
%count weight
lnpi=psi(post_Alpha1);
normterm=psi(sum(post_Alpha1,2));
normterm=repmat(normterm,1,nScores);
lnpi=lnpi-normterm;
indx = sub2ind([nScores nAgents], inputC{3}, inputC{1});
lnPiIndx = lnpi(:,indx);
weight(t,:)=exp(lnPiIndx(2,end-nAgents+1:end));   %ln_pi^k_(j=1), 
weight(t,:)=weight(t,:)/abs(sum(weight(t,:)));
% calculate performance
if round(post_T1(end)+1)~=target(t),
    Count(1) = Count(1)+1;      %PER
    if round(post_T1(end)+1)==1,
        Count(2) = Count(2) + 1;%FAR
    elseif round(post_T1(end)+1)==2,
        Count(3) = Count(3) + 1;%MAR
    end
end
    
        
IBCC_predict(t)=post_T1(end);
%[ Count, Counting ] = eval_perform( Count, post_T1, IBCC_target, predict_output', t, Counting, nAgents);
%post_T1(end)
end
for i=1:length(w),
    %should include a dynamic 3D array to indicate which parition i belongs
    %to. 
    part=partition(test_ind(i),F_EL);
    weightobs(1,i)=w{i}(part(1),part(2),part(3),1);
    weightobs(2,i)=w{i}(part(1),part(2),part(3),2);
    weightobs(3,i)=w{i}(part(1),part(2),part(3),3);
    weightobs(4,i)=w{i}(part(1),part(2),part(3),4);
    if nAgents>4,
        weightobs(5,i)=w{i}(part(1),part(2),part(3),5);
    end
end
%we may need to know the boudary, since the weight are assigned to the
%instead of features accessible to CEL, use the most relevant features

%% Combine Score and Plot ROC
score3=[score(test_ind,1:2)+1 score(test_ind,3:4) CELscore(:,end) (IBCC_predict(test_ind)+1)'];
%score3=[score(test_ind,1:2)+1 score(test_ind,3:4) CELscore(:,end)];
auc2=[];
for i=1:6,
[x2,y2,~,auc2(i)] = perfcurve(ytest,score3(:,i),2);
figure(49)
plot(x2,y2)
xlabel('true positive rate')
ylabel('false positive rate')
hold on
end
legend('logistic', 'SVM', 'RandomForest','LASSO','CEL', 'IBCC')
auc(kaui,:)=auc2;
celprediction(:,kaui)=CELscore(:,end);
logweight(:,kaui)=weightobs(1,:)';

saveas(gcf,'ROC_all.png')
%% Plot IBCC
k1=1;
k2=2;
k3=3;
for i=1:nAgents,
    figure(nAgents+i)
    scatter3(features(test_ind,k1),features(test_ind,k2),features(test_ind,k3),40,weight(test_ind,i),'filled')    % draw the scatter plot
    hold on
    ax = gca;
    ax.XDir = 'reverse';
    view(-31,14)

    xlabel(Feat_nam{k1})
    ylabel(Feat_nam{k2})
    zlabel(Feat_nam{k3})
    cb = colorbar;                                     % create and label the colorbar
    cb.Label.String = 'Weighting in IBCC';
    if strat_arg==true,
        title (FN_nam{i})
        saveas(gcf,['IBCC_',FN_nam{i},'.png'])
    else
        title(ML_nam{i})
        saveas(gcf,['IBCC_',ML_nam{i},'.png'])
    end
end

%% Plot CEL


for j=1:nAgents,
    
    figure(j)
    scatter3(features(test_ind,k1),features(test_ind,k2),features(test_ind,k3),40,weightobs(j,:),'filled')    % draw the scatter plot
    hold on
    ax = gca;
    ax.XDir = 'reverse';
    view(-31,14)
    xlabel(Feat_nam{k1})
    ylabel(Feat_nam{k2})
    zlabel(Feat_nam{k3})
    cb = colorbar;                                     % create and label the colorbar
    cb.Label.String = 'Weighting in CEL';
    if strat_arg==true,
        title (FN_nam{j})
        saveas(gcf,['CEL_',FN_nam{j},'.png'])
    else
        title(ML_nam{j})
        saveas(gcf,['CEL_',ML_nam{j},'.png'])
    end
end
%% less datapoint, time indexed
j=1;
figure(50)
txt=[1:sum(mod(test_ind,4)==0)]';
txts=cellstr(num2str(txt));
dx=0.01;dy=0.01;dz=0.01;
scatter3(features(test_ind(mod(test_ind,4)==0),k1),features(test_ind(mod(test_ind,4)==0),k2),features(test_ind(mod(test_ind,4)==0),k3),[1:sum(mod(test_ind,4)==0)]*5,weightobs(j,mod(test_ind,4)==0),'filled')
text(features(test_ind(mod(test_ind,4)==0),k1)+dx,features(test_ind(mod(test_ind,4)==0),k2)+dy,features(test_ind(mod(test_ind,4)==0),k3)+dz,txts);
hold on
ax = gca;
ax.XDir = 'reverse';
view(-31,14)

xlabel(Feat_nam{k1})
ylabel(Feat_nam{k2})
zlabel(Feat_nam{k3})

cb = colorbar;                                     % create and label the colorbar
cb.Label.String = 'Weighting in CEL';
if strat_arg==true,
    title (FN_nam{j})
    saveas(gcf,['TimedCEL_',FN_nam{j},'.png'])
else
    title(ML_nam{j})
    saveas(gcf,['TimedCEL_',ML_nam{j},'.png'])
end
%features=[shp, macd, vlt, tbill_dl(1:dtnum), rtn_past, existed];
