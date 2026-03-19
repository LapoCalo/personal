%% =========================================================
%  Project 1 — Single-Axis Satellite Attitude Control (PID)
%  
%  System:  G(s) = 1 / (I*s^2)   [double integrator]
%  Control: PID : Kp, Ki, Kd
%
%  Demonstrates:
%    1. Open-loop instability
%    2. Closed-loop step response
%    3. Stability margins (Bode)
%    4. Disturbance rejection
% =========================================================
clear; clc; close all;

%% --- Parameters ---
I        = 100;           % Moment of inertia [kg·m²]
theta_ref = 10*pi/180;   % Reference attitude: 10 deg [rad]
tau_dist  = 0.01;         % Disturbance torque [N·m]
t         = 0:0.01:60;   % Simulation time [s]

%% --- Plant ---
G = tf(1, [I 0 0]);

%% --- PID gains (pole placement) ---
% Target: wn = 0.5 rad/s, zeta = 0.7
wn   = 0.5;
zeta = 0.7;

Kp = I * wn^2;
Ki = 0.1;
Kd = I * 2*zeta*wn;

fprintf('=== PID Gains ===\n');
fprintf('Kp = %.3f\nKi = %.4f\nKd = %.3f\n\n', Kp, Ki, Kd);

C = pid(Kp, Ki, Kd);

%% --- Closed-loop transfer functions ---
T_ref  = feedback(C*G, 1);      % Reference tracking
T_dist = feedback(G, C);         % Disturbance rejection

%% =========================================================
%  FIGURE 1 — Open Loop vs Closed Loop
% =========================================================
figure('Name','1 : Open Loop vs Closed Loop','NumberTitle','off',...
       'Position',[100 100 900 600]);

% Open loop: constant torque -> attitude drifts as ramp
subplot(2,2,1)
[y_ol, t_ol] = step(tau_dist * G, 0:0.01:10);
plot(t_ol, y_ol*180/pi, 'r', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Open Loop : attitude drift under disturbance');
grid on;
annotation('textbox',[.12 .72 .35 .06], ...
    'String','No control: attitude drifts unboundedly', ...
    'EdgeColor','r','BackgroundColor','#fff0f0','FontSize',8);

% Closed loop step response
subplot(2,2,2)
[y_cl, t_cl] = step(theta_ref * T_ref, t);
plot(t_cl, y_cl*180/pi, 'b', 'LineWidth', 2); hold on;
yline(theta_ref*180/pi, 'r--', 'LineWidth',1.5, 'Label','Reference (10°)');
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Closed Loop — step response');
grid on;

% Disturbance rejection
subplot(2,2,3)
[y_d_ol, ~] = step(tau_dist * G,      t);
[y_d_cl, ~] = step(tau_dist * T_dist, t);
plot(t, y_d_ol*180/pi, 'r--', 'LineWidth',1.5, ...
     'DisplayName','Open loop (drift)'); hold on;
plot(t, y_d_cl*180/pi, 'b',   'LineWidth',2,   ...
     'DisplayName','Closed loop (rejected)');
xlabel('Time [s]'); ylabel('\Delta\theta [deg]');
title('Disturbance rejection (0.01 N·m)');
legend; grid on;

% Effect of Kd — compare PD only vs PID
subplot(2,2,4)
gains = [Kp, 0,   0;    % P only
         Kp, 0,   Kd;   % PD
         Kp, Ki,  Kd];  % PID
labels = {'P only','PD','PID'};
colors = {'r','orange','b'};
hold on;
for i = 1:3
    Ci = pid(gains(i,1), gains(i,2), gains(i,3));
    Ti = feedback(Ci*G, 1);
    [yi, ~] = step(theta_ref * Ti, t);
    plot(t, yi*180/pi, 'LineWidth', 1.8, 'DisplayName', labels{i});
end
yline(theta_ref*180/pi, 'k--', 'LineWidth',1);
xlabel('Time [s]'); ylabel('\theta [deg]');
title('Effect of PID terms');
legend; grid on;

%% =========================================================
%  FIGURE 2 — Stability Analysis
% =========================================================
figure('Name','2 — Stability Margins','NumberTitle','off',...
       'Position',[100 100 900 400]);

subplot(1,2,1)
margin(C*G);
title('Bode : Open loop with PID');
grid on;

subplot(1,2,2)
rlocus(C*G);
title('Root Locus');
grid on;

%% --- Performance metrics ---
info = stepinfo(theta_ref * T_ref);
[Gm, Pm] = margin(C*G); 

fprintf('=== Closed-Loop Performance ===\n');
fprintf('Rise Time:      %.2f s\n',  info.RiseTime);
fprintf('Settling Time:  %.2f s\n',  info.SettlingTime);
fprintf('Overshoot:      %.2f %%\n', info.Overshoot);
fprintf('Phase Margin:   %.1f deg\n', Pm);
fprintf('Gain Margin:    %.1f dB\n',  20*log10(Gm));