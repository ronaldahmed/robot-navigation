function [mu, P] = compute_gaussian_from_samples(X)
% COMPUTE_GAUSSIAN_FROM_SAMPLES Compute gaussian approximation
%
%   [MU, P] = COMPUTE_GAUSSIAN_FROM_PARTICLES(X) computes the gaussian 
%             statistics of a set of samples.
%
% OUTPUTS
%   mu: Mean of the samples. (column)
%    P: Covarianze of the samples
%
% INPUTS
%   X: Set of samples in which each row is a sample. 


%----------------------------------------------------------------------%
% 1. INITIAL CONFIGURATION
%----------------------------------------------------------------------%
%  1.1. Number of samples
N = size(X,1);

%  1.2. Dimensions of the samples
D = size(X,2);


%----------------------------------------------------------------------%
% 2. COMPUTE THE MEAN
%----------------------------------------------------------------------%
mu = mean(X,1);     % Mean [1x3]


%----------------------------------------------------------------------%
% 3. COMPUTE THE COVARIANCE
%----------------------------------------------------------------------%
%  3.1. USEFUL TERM
M = X - ones(N,1)*mu;

%  3.2. COMPUTE THE COVARIANCE
P = zeros(D);
for n = 1:N
    P = P + M(n,:)'*M(n,:);
end


%----------------------------------------------------------------------%
% 4. FINAL CONFIGURATION
%----------------------------------------------------------------------%
%  4.1. RETURN UNBIASED COVARIANCE
P = (1/(N-1))*P;

%  4.2. RETURN THE MEAN AS A COLUMN
mu = mu';


end