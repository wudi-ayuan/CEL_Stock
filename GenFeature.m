function [ features, target, train_ind,test_ind ] = GenFeature()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%feature list: 

% 1.macd
% 2.vlt, 
% 3.rtn_past, 
% 4.dma_5
% 5.dma_15
% 6.dma_25
% 7.RDP_5
% 8.RDP_15
% 9.RDP_25
% 10.VDP_5
% 11.VDP_15
% 12.VDP_25
% 13.RSI
% 14.delta Volum
% 15.dma_20
% 16.bollinger
% 17.minmax
% rawdata_inex 1008: 01/02/2015
    %initialize 

    sp500=importdata('S&P500.csv');
    space=26;        % window size
    dayback=5;
    raw_data=sp500.data;
    raw_data=raw_data(1:end-1,:);  %ends at one day before the last day
    dtnum=size(raw_data,1)-space;
    
    target_rtn=(sp500.data(2:end,end)-sp500.data(1:end-1,end))./sp500.data(1:end-1,end);
    target(:,1)=target_rtn(end-dtnum+1:end);
    
    %%Temporal features.  
    %volatility 
    vlt=zeros(dtnum,1);
    %daily return(yesterday)
    rtn_dl=(raw_data(2:end,end)-raw_data(1:end-1,end))./raw_data(1:end-1,end);

    %50 day simple MA
    dma=zeros(dtnum,3);
    %MACD
    macd=ema(raw_data(:,end),12)-ema(raw_data(:,end),26);
    macd=macd(end-dtnum+1:end);
    %RSI
    U=max(0,raw_data(2:end,end)-raw_data(1:end-1,end));
    D=min(0,raw_data(2:end,end)-raw_data(1:end-1,end));
    rs=ema(U,10)./ema(D,10);
    rsi=100-100./(1+rs);
    rsi=rsi;
    rsi=rsi(end-dtnum+1:end);
    %bollinger Bands
    blgbnd=zeros(dtnum,1);
    %MinMax
    mnx=zeros(dtnum,1);
    %RDP -5-10-15
    RDP=zeros(dtnum,3);
    %VDP -5-10-15
    RDP=zeros(dtnum,3);
    %moving avg -5 -15 -25
    

    
    %existed feat;'Open'    'High'    'Low'    'Close'    'Volume'    'Adj Close' 
    existed=raw_data(end-dtnum-1:end,:);
    vlm_dlt=raw_data(2:2+dtnum-1,5)-raw_data(3:3+dtnum-1,5);

    for i=1:dtnum, 
        %volatility, tval
        vlt(i) = std(rtn_dl(i:dayback+i-1));
        %5 day simple MA
        dma(i,1) = mean(raw_data(space+i-5+1:space+i,end));
        dma(i,2) = mean(raw_data(space+i-15+1:space+i,end));
        dma(i,3) = mean(raw_data(space+i-25+1:space+i,end));
        %RDP
        RDP(i,1) = (raw_data(space+i,end)-raw_data(space+i-5,end))/raw_data(space+i-5,end);
        RDP(i,2) = (raw_data(space+i,end)-raw_data(space+i-15,end))/raw_data(space+i-15,end);
        RDP(i,3) = (raw_data(space+i,end)-raw_data(space+i-25,end))/raw_data(space+i-25,end);
        %VDP
        VDP(i,1) = (raw_data(space+i,end-1)-raw_data(space+i-5,end-1))/raw_data(space+i-5,end-1);
        VDP(i,2) = (raw_data(space+i,end-1)-raw_data(space+i-15,end-1))/raw_data(space+i-15,end-1);
        VDP(i,3) = (raw_data(space+i,end-1)-raw_data(space+i-25,end-1))/raw_data(space+i-25,end-1);


        %bollinger Bands
        dma20(i) = mean(raw_data(space+i-20+1:space+i,end));
        std_dma20(i)= std(raw_data(space+i-20+1:space+i,end));
        UPbnd(i)=dma20(i)+2*std_dma20(i);
        LPbnd(i)=dma20(i)-2*std_dma20(i);
        blgbnd(i) = (raw_data(i+1,end)-LPbnd(i))/(UPbnd(i)-LPbnd(i));
        blgbnd(i)=blgbnd(i)>0.5;
        %MinMax
        L=20;
        if raw_data(i,end)>max(raw_data(space+i-L+1:space+i,end)),
            mnx(i)=1;
        elseif raw_data(i,end)<min(raw_data(space+i-L+1:space+i,end)),
            mnx(i)=0;
        else
            mnx(i)=rand(1)>0.5;
        end
    end
    

    rtn_past=rtn_dl(end-dtnum+1:end);
    features=[ macd, vlt, rtn_past, dma, RDP, VDP, rsi, vlm_dlt, dma20', blgbnd,mnx];
    %normalize
    for i=1:size(features,2)-5,
        features(:,i)=(features(:,i)-mean(features(:,i)))/(max(features(:,i))-min(features(:,i)));
    end
    target(:,1)=target(:,1);
    vol=raw_data(end-dtnum+1:end,5);
    target(:,2)=(vol-mean(vol)/(max(vol)-min(vol)))/10^9;
    target(:,3)=target(:,1)>0;
    %%test/ train = 1:4
    for i=1:size(sp500.textdata,1),
        if strcmp(sp500.textdata{i,1},'1/2/2015');
            test_indstart=i-1;
        end
    end
    test_ind=test_indstart-space;
    test_ind=test_ind:size(features,1);
    train_ind=1:test_ind-1;
    %hist(features(1:test_ind,1:12))
end

