// 交互式功能脚本
document.addEventListener('DOMContentLoaded', function() {
    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // 标签页切换
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // 移除所有活动状态
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // 添加当前活动状态
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // 图表初始化
    initCharts();
    
    // 动画效果
    animateOnScroll();
});

// 初始化图表
function initCharts() {
    // 特征重要性图表
    const featureImportanceCtx = document.getElementById('featureImportanceChart');
    if (featureImportanceCtx) {
        new Chart(featureImportanceCtx, {
            type: 'bar',
            data: {
                labels: ['EBIT/EV', 'P/B倒数', 'FCF/Price', 'ROE', '资产负债率', '毛利率稳定性', '6个月动量', 'RSI', '均线交叉', '波动率倒数', '贝塔倒数', '回撤倒数'],
                datasets: [{
                    label: '特征重要性',
                    data: [0.18, 0.12, 0.09, 0.15, 0.07, 0.08, 0.11, 0.05, 0.04, 0.06, 0.03, 0.02],
                    backgroundColor: [
                        'rgba(52, 152, 219, 0.7)',
                        'rgba(52, 152, 219, 0.7)',
                        'rgba(52, 152, 219, 0.7)',
                        'rgba(46, 204, 113, 0.7)',
                        'rgba(46, 204, 113, 0.7)',
                        'rgba(46, 204, 113, 0.7)',
                        'rgba(155, 89, 182, 0.7)',
                        'rgba(155, 89, 182, 0.7)',
                        'rgba(155, 89, 182, 0.7)',
                        'rgba(241, 196, 15, 0.7)',
                        'rgba(241, 196, 15, 0.7)',
                        'rgba(241, 196, 15, 0.7)'
                    ],
                    borderColor: [
                        'rgba(52, 152, 219, 1)',
                        'rgba(52, 152, 219, 1)',
                        'rgba(52, 152, 219, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(46, 204, 113, 1)',
                        'rgba(155, 89, 182, 1)',
                        'rgba(155, 89, 182, 1)',
                        'rgba(155, 89, 182, 1)',
                        'rgba(241, 196, 15, 1)',
                        'rgba(241, 196, 15, 1)',
                        'rgba(241, 196, 15, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '重要性得分'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '因子'
                        },
                        ticks: {
                            autoSkip: false,
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: '因子重要性排名'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // 市场仓位变化图表
    const positionCtx = document.getElementById('positionChart');
    if (positionCtx) {
        // 生成日期数组
        const dates = [];
        const startDate = new Date('2023-01-01');
        const endDate = new Date('2024-12-31');
        for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 15)) {
            dates.push(new Date(d).toISOString().split('T')[0]);
        }
        
        // 生成仓位数据
        const positionData = [];
        for (let i = 0; i < dates.length; i++) {
            // 模拟不同市场环境下的仓位变化
            if (i < dates.length * 0.2) {
                // 初始阶段，中等仓位
                positionData.push(60 + Math.random() * 20);
            } else if (i < dates.length * 0.4) {
                // 市场上涨，高仓位
                positionData.push(80 + Math.random() * 20);
            } else if (i < dates.length * 0.6) {
                // 市场调整，降低仓位
                positionData.push(40 + Math.random() * 20);
            } else if (i < dates.length * 0.8) {
                // 市场反弹，增加仓位
                positionData.push(70 + Math.random() * 20);
            } else {
                // 最后阶段，高仓位
                positionData.push(90 + Math.random() * 10);
            }
        }
        
        new Chart(positionCtx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '市场仓位 (%)',
                    data: positionData,
                    borderColor: 'rgba(231, 76, 60, 1)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: '仓位百分比 (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '日期'
                        },
                        ticks: {
                            maxTicksLimit: 8
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'RSRS择时策略仓位变化'
                    }
                }
            }
        });
    }
    
    // 年度收益对比图表
    const yearlyReturnsCtx = document.getElementById('yearlyReturnsChart');
    if (yearlyReturnsCtx) {
        new Chart(yearlyReturnsCtx, {
            type: 'bar',
            data: {
                labels: ['2023', '2024'],
                datasets: [
                    {
                        label: '策略收益',
                        data: [5.62, 29.34],
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgba(46, 204, 113, 1)',
                        borderWidth: 1
                    },
                    {
                        label: '基准收益',
                        data: [3.21, 6.91],
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '收益率 (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '年份'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: '年度收益对比'
                    }
                }
            }
        });
    }
}

// 滚动动画效果
function animateOnScroll() {
    const elements = document.querySelectorAll('.feature, .factor-category, .process-step, .stat');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    elements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(element);
    });
}
