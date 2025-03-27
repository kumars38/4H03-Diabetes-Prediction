function [T,P,R2] = nipalspca_me(X,A)

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
    X_ctr(:,i) = X(:,i) - mean(X(:,i),"omitnan");
    X_CS(:,i) = X_ctr(:,i) / std(X_ctr(:,i),"omitnan");
end % for

% NIPALS algorithm

X_fit = X_CS;

for a = 1:A
    
    % Chosing 1st column of X_CS as initial guess for ta
    ta = X_CS(:,1);

    for i = 1:max_iters
        pa_T = ta'*X_fit/(ta'*ta); % Step 2.1 - Reg xk on ta
        % disp("pa_T")
        % disp(pa_T)
        pa = pa_T'/norm(pa_T);     % Step 2.2 - normalize p vector
        % pa = pa_T'/sum(pa_T.^2, "omitnan");     % Step 2.2 - normalize p vector
        % disp("pa")
        % disp(pa)

        ta_new = X_fit*pa/(pa'*pa);
        % disp("ta_new")
        % disp(ta_new)

        if sum((ta_new - ta).^2,"omitnan") <= tol
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