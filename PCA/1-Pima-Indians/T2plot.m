function fig = T2plot(T,N,A)

% Function for generating Hotelling's T2 plot
% Written by Steven Chui

% T - score matrix from PCA
% N - number of observation from data
% A - number of principle component fitteds

T2 = zeros(N,1);
for i = 1:A
    T2 = T2 + (T(:,i)/std(T(:,i))).^2;
end % for

% 95% T2 line
p = 0.95;
T2_95 = (N-1)*(N+1)*A/N/(N-A) * finv(p,A,N-A);
% disp("T2_95:")
% disp(T2_95)

% 99% T2 line
p = 0.99;
T2_99 = (N-1)*(N+1)*A/N/(N-A) * finv(p,A,N-A);
% disp("T2_99:")
% disp(T2_99)

% plot
xRange = 1:1:N;
fig = figure();
plot(xRange,T2,'-ko')
hold on
plot(xRange, ones(N,1)*T2_95, '--r')
plot(xRange, ones(N,1)*T2_99, '-r')
hold off
xlim([0 N]);
xlabel("Observation Number")
ylabel("Hotelling's T^2")
legend('T^2 Values','95% Limit','99% Limit')
grid on 
box on


end %function