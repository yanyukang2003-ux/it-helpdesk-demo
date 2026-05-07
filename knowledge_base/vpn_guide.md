# VPN 连接指南

## 什么是公司 VPN

公司 VPN（虚拟专用网络）用于在外部网络环境下安全访问公司内部资源，包括内网系统、文件服务器和开发环境。

## 安装 VPN 客户端

### Windows 系统
1. 访问 IT 自服务门户: https://it.company.com/vpn
2. 下载 GlobalProtect 客户端（Windows 版）
3. 双击安装包，按提示完成安装
4. 安装完成后重启电脑

### macOS 系统
1. 打开 App Store，搜索 "GlobalProtect"
2. 点击"获取"下载安装
3. 或者访问 IT 自服务门户下载 dmg 安装包

## 连接 VPN

1. 打开 GlobalProtect 客户端
2. 在服务器地址栏输入: vpn.company.com
3. 点击"连接"
4. 输入你的公司邮箱账号和密码
5. 完成手机上的二次验证（MFA）
6. 连接成功后，任务栏会显示绿色盾牌图标

## 常见问题

### VPN 连接失败
- 确认网络连接正常（能否打开百度）
- 检查账号密码是否正确
- 确认 MFA 验证是否超时（有效期 60 秒）
- 尝试重启 VPN 客户端
- 如仍无法连接，请提交工单

### VPN 连接频繁断开
- 检查网络是否稳定
- 关闭其他占用大量带宽的应用
- 尝试切换到有线网络
- 更新 VPN 客户端到最新版本

### VPN 连接后无法访问内网
- 确认 VPN 状态为"已连接"
- 尝试断开后重新连接
- 清除 DNS 缓存: Windows 使用 ipconfig /flushdns，macOS 使用 sudo dscacheutil -flushcache
