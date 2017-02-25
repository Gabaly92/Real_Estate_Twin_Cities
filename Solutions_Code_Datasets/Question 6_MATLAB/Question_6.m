clc; clear; close all;
data_sheet2=xlsread('Dataset.xlsx',2);
[num,str,raw]=xlsread('Dataset.xlsx',1);
zipcode=data_sheet2(:,1);
population=data_sheet2(:,2);
twincity_zipcode=num(:,5);
listing=zeros(33092,1);
list_per_pop=zeros(33092,1);
for i=1:33092
    current_zip=zipcode(i,1);
    for j=1:1111
        if(current_zip==twincity_zipcode(j,1))
            listing(i,1)=listing(i,1)+1;
        end
    end
end
for i=1:33092
    if(population(i,1)==0)
        list_per_pop(i,1)=0;
    else
    list_per_pop(i,1)=listing(i,1)/population(i,1);
    end
end
[B,I] = sort(list_per_pop,'descend') ;
required_zip=zeros(10,1);
%part a result
for i=1:10
    required_zip(i,1)=zipcode(I(i,1),1);
end
%part b result

ratio=zeros(1111,1);
for i=1:1111
    current_zip=twincity_zipcode(i,1);
    for j=1:33092
        if(current_zip==data_sheet2(j,1))
           cur_population=data_sheet2(j,2);
        end
    end
    ratio(i,1)=num(i,6)/cur_population;
end
[max_listprice_pop,index] = sort(ratio,'descend') ;
list_zip=num(index(1,1),5);
display('top 10 zip codes with highest amount of listing per population size');
required_zip
display('zip code with highest listing price per person');
list_zip

% Comments on Zip Code with highest listing price per person:
% After Googling this zip code, I found that the c;ass of the people in
% this zip code are mostly middle high class, which explains the price of
% this zip code
    
    
    

