function [vsel,crbvar,time_elaspsed,beta,accum_load,T] = fcovsel(X,Y,lvs)
% COVSEL without predictor matrix deflation
Xorig = X;  % saving a copy of X matrix for estimaiton of regression coefficient
Yorig = Y;  % saving a copy of response to estimate the offset for regression coefficient   
X = X-mean(X); % mean centering of predictor
Y = Y-mean(Y); % mean centering of response
Q = zeros(size(Y,2),lvs); % pre-allocation for coefficient of Y with respect to T
T = zeros(size(Y,1),lvs); % pre-allocation for scores
beta = cell(size(Y,2)); % pre-allocation for betas
crbvar = zeros(lvs,1); % pre-allocation for variation explained in Y
vartot = sum(sum(Y.*Y)); % total variation in response Y
vsel = zeros(1,lvs); % pre-allocation for storng selected variables
tic; 
accum_load = zeros(size(X,2),lvs);
for i = 1:lvs  % loop for selecting variables
    V = X'*Y;  % estimation of covariance
    accum_load(:,i) = V;
    [~,vsel(1,i)] = max(sum(abs(V),2)); % % Variable of maximum covariance with Y
    t = X(:,vsel(1,i)); % scores based on loading weight
    if i>1
        t = t - T(:,1:i-1)*(T(:,1:i-1)'*t); % orthgonalized wrt previous T-scores
    end
    T(:,i) = t/norm(t); % Normalize the score 
    Q(:,i)= Y'*T(:,i); % Regression coeff wrt T(:,i)
    Y = Y - T(:,i)*(T(:,i)'*Y); % Calculate Y-residuals
    crbvar(i,1) = (vartot - sum(sum(Y.*Y)))/vartot; % Estimate explained variance 
end
time_elaspsed = toc;
%%%%%%%%%%%%%%%%%% post-processing for regression vector %%%%%%%%%%%%%%%%%%%%%%%%%
Pb = X(:,vsel)'*T; % X-loadings
PtW = triu(Pb'*eye(lvs)); % The W-coordinates of (the projected) P
for i =1:size(Y,2) % The X-regression coefficients and intercepts for responses
    beta{i}  = cumsum(bsxfun(@times,eye(lvs)/PtW, Q(i,:)),2);
    beta{i}  = [mean(Yorig(:,i)) - mean(Xorig(:,vsel))*beta{i}(:,end); beta{i}(:,end)];
end
end
