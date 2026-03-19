%% =========================================================
%  Project 2 — 3-Axis Attitude Control (Quaternion PD)
%  Correct approach: error computed in body frame
%  No growing reference quaternion
% =========================================================
clear; clc; close all;

%% --- Parameters ---
Ix = 100; Iy = 120; Iz = 80;
I  = [Ix; Iy; Iz];

mu = 3.986e14;
r = (6371 + 500) * 1e3;
omega_orb = sqrt(mu / r^3);

wn   = 0.3;
zeta = 0.7;
Kp   = I * wn^2;
Kv   = I * 2*zeta*wn;

req_deg  = 0.1;
err0_rad = 5 * pi/180;

%% --- Initial conditions ---
% Satellite starts with a 5 deg pointing error about X.
% State: attitude error quaternion q_e and body angular velocity om.
% When q_e = [1 0 0 0], the satellite is perfectly pointed.
q_e = [cos(err0_rad/2), sin(err0_rad/2), 0, 0];
om  = [0; 0; 0]; % zero angular velocity at the beginning

%% --- Simulation ---
Ts = 0.01;
T  = 300;
N  = round(T / Ts);

log_t   = zeros(N,1); %pre-allocation to make the for faster
log_err = zeros(N,1);
log_tau = zeros(N,3);
log_om  = zeros(N,3);
log_qe  = zeros(N,4);

for k = 1:N

    %% Error signal — directly from error quaternion
    % q_e(1) = scalar part (w), q_e(2:4) = vector part
    % Shortest path: ensure w >= 0
    if q_e(1) < 0; q_e = -q_e; end
    e_vec = q_e(2:4)';   % [3x1] attitude error signal

    %% Control law — quaternion PD
    % omega_ref = orbital rate about Y axis (nadir pointing)
    om_ref = [0; -omega_orb; 0];
    tau    = -Kp .* e_vec - Kv .* (om - om_ref);

    %% Dynamics — Euler equations
    dom = [(tau(1) - (Iz-Iy)*om(2)*om(3)) / Ix;
           (tau(2) - (Ix-Iz)*om(3)*om(1)) / Iy;
           (tau(3) - (Iy-Ix)*om(1)*om(2)) / Iz];

    %% Error quaternion kinematics
    % q_e_dot = 0.5 * q_e * [0; omega_error]
    % where omega_error = om - om_ref expressed in body frame
    om_err = om - om_ref;
    Omg = [ 0         -om_err(1) -om_err(2) -om_err(3);
            om_err(1)  0          om_err(3) -om_err(2);
            om_err(2) -om_err(3)  0          om_err(1);
            om_err(3)  om_err(2) -om_err(1)  0        ];
    dq_e = (0.5 * Omg * q_e')';

    %% Integration
    om  = om  + dom  * Ts;
    q_e = q_e + dq_e * Ts;
    q_e = quatnormalize(q_e);

    %% Log
    log_t(k)    = k * Ts;
    log_err(k)  = 2*asin(min(norm(e_vec),1)) * 180/pi;
    log_tau(k,:)= tau';
    log_om(k,:) = om' * 180/pi;
    log_qe(k,:) = q_e;
end

%% --- Results ---
settled = find(log_err < req_deg, 1, 'first');
if ~isempty(settled)
    fprintf('Settling time:      %.1f s\n',   log_t(settled));
    fprintf('Steady-state error: %.4f deg\n', mean(log_err(end-1000:end)));
    fprintf('Requirement MET\n\n');
else
    fprintf('Requirement NOT met\n\n');
end

%% =========================================================
%  FIGURE 1 — Performance
% =========================================================
figure('Name','3-Axis AOCS — Performance','NumberTitle','off',...
       'Position',[100 80 1000 600]);

subplot(2,2,1)
plot(log_t, log_err, 'b', 'LineWidth', 2); hold on;
yline(req_deg,'r--','LineWidth',1.5,'Label','Requirement (0.1°)');
if ~isempty(settled)
    xline(log_t(settled),'g--','LineWidth',1.2,...
          'Label',sprintf('Settled: %.0f s', log_t(settled)));
end
xlabel('Time [s]'); ylabel('Error [deg]');
title('Pointing Error (3-axis norm)');
grid on;

subplot(2,2,2)
plot(log_t, log_tau, 'LineWidth', 1.8);
xlabel('Time [s]'); ylabel('\tau [N·m]');
title('Commanded Torque — 3 axes');
legend('X','Y','Z'); grid on;

subplot(2,2,3)
plot(log_t, log_om, 'LineWidth', 1.8);
xlabel('Time [s]'); ylabel('\omega [deg/s]');
title('Angular Velocity — 3 axes');
legend('X','Y','Z'); grid on;

subplot(2,2,4)
plot(log_t, log_qe, 'LineWidth', 1.8);
xlabel('Time [s]'); ylabel('Component [-]');
title('Error quaternion components');
legend('w','x','y','z'); grid on;

%% =========================================================
%  FIGURE 2 — Gain Sensitivity
% =========================================================
figure('Name','Gain Sensitivity','NumberTitle','off',...
       'Position',[100 80 900 400]);

subplot(1,2,1); hold on;
wn_values = [0.1, 0.3, 0.5, 0.8];
for i = 1:length(wn_values)
    err_i = run_sim(wn_values(i), zeta, Ix, Iy, Iz, ...
                    omega_orb, err0_rad, N, Ts);
    plot(log_t, err_i, 'LineWidth', 1.8, ...
         'DisplayName', sprintf('\\omega_n=%.1f', wn_values(i)));
end
yline(req_deg,'k--','LineWidth',1.5,'Label','Requirement');
xlabel('Time [s]'); ylabel('Error [deg]');
title('Effect of \omega_n on settling');
legend('Location','northeast'); grid on;
xlim([0 150]); ylim([0 6]);

subplot(1,2,2); hold on;
zeta_values = [0.3, 0.5, 0.7, 1.0];
for i = 1:length(zeta_values)
    err_i = run_sim(wn, zeta_values(i), Ix, Iy, Iz, ...
                    omega_orb, err0_rad, N, Ts);
    plot(log_t, err_i, 'LineWidth', 1.8, ...
         'DisplayName', sprintf('\\zeta=%.1f', zeta_values(i)));
end
yline(req_deg,'k--','LineWidth',1.5,'Label','Requirement');
xlabel('Time [s]'); ylabel('Error [deg]');
title('Effect of \zeta on damping');
legend('Location','northeast'); grid on;
xlim([0 150]); ylim([0 6]);

%% =========================================================
%  LOCAL FUNCTIONS
% =========================================================

function err_log = run_sim(wni, zi, Ix, Iy, Iz, ...
                            omega_orb, err0_rad, N, Ts)
%RUN_SIM  Fresh simulation returning pointing error [deg].

    I      = [Ix; Iy; Iz];
    Kp     = I * wni^2;
    Kv     = I * 2*zi*wni;
    om_ref = [0; -omega_orb; 0];

    q_e    = [cos(err0_rad/2), sin(err0_rad/2), 0, 0];
    om     = [0; 0; 0];

    err_log = zeros(N,1);

    for k = 1:N
        if q_e(1) < 0; q_e = -q_e; end
        e_vec  = q_e(2:4)';
        tau    = -Kp .* e_vec - Kv .* (om - om_ref);

        dom = [(tau(1)-(Iz-Iy)*om(2)*om(3))/Ix;
               (tau(2)-(Ix-Iz)*om(3)*om(1))/Iy;
               (tau(3)-(Iy-Ix)*om(1)*om(2))/Iz];

        om_err = om - om_ref;
        Omg = [ 0         -om_err(1) -om_err(2) -om_err(3);
                om_err(1)  0          om_err(3) -om_err(2);
                om_err(2) -om_err(3)  0          om_err(1);
                om_err(3)  om_err(2) -om_err(1)  0        ];

        q_e = q_e + (0.5*Omg*q_e')'*Ts;
        q_e = quatnormalize(q_e);
        om  = om  + dom*Ts;

        err_log(k) = 2*asin(min(norm(e_vec),1))*180/pi;
    end
end