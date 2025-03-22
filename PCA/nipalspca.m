function [T,P,R2] = nipalspca(X,A)

N = size(X,1);
K = size(X,2);
tol = 1e-8;
max_iters = 500;

% Placeholders
T = zeros(N,A);
P = zeros(K,A);
R2 = zeros(1,A);
X_ctr = zeros(N,K);
X_CS = zeros(N,K);

% Centering and scaling data
for i = 1:K
    X_ctr(:,i) = X(:,i) - mean(X(:,i));
    X_CS(:,i) = X_ctr(:,i) / std(X_ctr(:,i));
end % for

% NIPALS algorithm

X_fit = X_CS;

for a = 1:A
    
    % Chosing 1st column of X_CS as initial guess for ta
    ta = X_CS(:,1);

    for i = 1:max_iters
        pa_T = ta'*X_fit/(ta'*ta); % Step 2.1 - Reg xk on ta
        pa = pa_T'/norm(pa_T);     % Step 2.2 - normalize p vector

        ta_new = X_fit*pa/(pa'*pa);

        if norm(ta_new - ta) <= tol
            T(:,a) = ta_new;   % Save ta_new into T
            P(:,a) = pa;      % Save pa into P
            break
        end % if

        % Prep for next iter
        ta = ta_new;

    end % for i
    
    % Step 3: Deflating X_fit
    Xa_hat = T(:,a)*P(:,a)';
    Ea = X_fit - Xa_hat;

    % Calculate R2
    R2(a) = 1 - var(Ea)/var(X_CS);

    % For next iteration
    X_fit = Ea;

end % for a

end % function