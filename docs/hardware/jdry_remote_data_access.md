# JDry-250-QT Remote Data Access

This document records the working, read-only procedure used on 2026-08-09 to
retrieve refrigerator data from the JanisULT/FormFactor JDry-250-QT. It does
not require the JaSoftDry HMI to be installed on the remote computer and does
not provide remote control of the refrigerator.

## Network layout

| Device or interface | Address | Purpose |
| --- | --- | --- |
| cRIO controller | `172.16.0.10` | Stores the refrigerator data logs and provides WebDAV on port 80 |
| Built-in HMI touchscreen | `172.16.0.20` | Runs JaSoftDry HMI on the private GHS network |
| JetWay private interface | `172.16.0.254` | Gateway between the private GHS network and the laboratory LAN |
| JetWay laboratory interface | `192.168.88.248` | SSH endpoint reached from the data-analysis computer |
| JetWay laboratory gateway | `192.168.88.1` | Gateway for the `192.168.88.0/24` network |

The laboratory Ethernet cable must be connected to the GHS rear-panel port
labelled `J7-RJ45 Remote Access`. The `J6-RJ45 Maint.` port is not used for
this procedure.

## Prerequisites

- The remote Windows computer must be on the `192.168.88.0/24` laboratory
  network, or connected through an institutional VPN that can reach it.
- PuTTY must be installed on the remote Windows computer.
- The current JetWay SSH credentials must be available from the lab
  administrator or FormFactor. Do not store the password in this repository.
- Data logging must have been started from the built-in HMI if new log files
  are required.

## Verify the JetWay identity

The SSH host-key fingerprint observed on 2026-08-09 was:

```text
ssh-ed25519 255 SHA256:Sd8tNeE5ujoNsfWAmm6P5MZ+YbCese3WZDWQ6mnmeqg
```

When PuTTY displays a first-connection security alert, compare its fingerprint
with a trusted value before accepting it. The fingerprint can be checked from
an authenticated JetWay shell with:

```sh
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Do not continue if the fingerprints differ unexpectedly.

## Configure PuTTY

1. Open PuTTY on the remote Windows computer.
2. On the **Session** page, enter:

   ```text
   Host Name: 192.168.88.248
   Port: 22
   Connection type: SSH
   ```

3. Open **Connection -> SSH -> Tunnels**.
4. Add the following local port forward:

   ```text
   Source port: 80
   Destination: 172.16.0.10:80
   Type: Local
   Address family: Auto
   ```

5. Click **Add** and confirm that the forwarded-ports list contains an entry
   equivalent to:

   ```text
   L80 172.16.0.10:80
   ```

6. Return to **Session**, save the configuration as `JDry data`, and click
   **Open**.
7. Authenticate with the current JetWay SSH account and keep the PuTTY window
   open. Closing it closes the tunnel.

The PuTTY Event Log should contain:

```text
Local port 80 forwarding to 172.16.0.10:80
```

## Retrieve the refrigerator logs

With PuTTY connected, first test the tunnel in a browser:

```text
http://localhost/files/
```

Then open File Explorer and enter:

```text
\\localhost\files
```

Authenticate to the cRIO using the current cRIO credentials. Navigate to the
verified WebDAV path:

```text
home\lvuser\natinst\LabVIEW Data
```

For a drive mapped as `Y:`, the complete path is:

```text
Y:\home\lvuser\natinst\LabVIEW Data
```

The similarly named `Y:\C\ni-rt` directory contains the real-time application
startup files and is not the data-log directory.

The timestamped `.xls` files contain recorded temperatures, resistances,
pressures, gas flow, and heater powers. Copy the required files to a normal
folder on the remote computer and analyze the copies rather than modifying the
files on the cRIO.

The cRIO has approximately 512 MB of storage. According to the operating
manual, logs grow by roughly 3-4 MB per day, so logs should be archived
regularly. Do not delete cRIO data until it has been copied and verified.

## Troubleshooting

### `Name or service not known`

The verified setup initially failed because the PuTTY tunnel destination had a
leading space. The Event Log showed:

```text
Forwarded connection refused by remote: Connect failed [Name or service not known]
```

Remove the tunnel entry and re-enter `172.16.0.10:80` with no leading or
trailing spaces.

### Cannot connect to `192.168.88.248`

Confirm that the remote computer is on the laboratory network or VPN and that
`J7-RJ45 Remote Access` is connected. Do not use `172.16.0.254` as the PuTTY
host from the remote computer; that address exists only inside the GHS private
network.

### Browser works but File Explorer does not

Use `\\localhost\files`, without `:80`, because port 80 is already the local
endpoint of the SSH tunnel. If the browser test succeeds but File Explorer
still fails, check that the Windows WebClient service is available before
changing the tunnel.

## Safety and security

- This procedure is for retrieving logs only; it does not forward the HMI
  control port.
- The JetWay runs an old Ubuntu release. Do not perform operating-system or
  package upgrades without FormFactor approval.
- Keep the J7 interface behind the institutional firewall or VPN.
- Do not commit passwords or other authentication secrets.
- Coordinate password or network changes with the lab administrator and
  FormFactor so the refrigerator's private control network is not disrupted.

## Source

The vendor procedure is described in *JDry-250-QT Operating Manual - 22112
Formfactor*, principally sections 11.1.2.6 and 11.2.
