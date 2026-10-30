import {test,expect} from '@playwright/test'
import {readFileSync,existsSync} from 'node:fs'
test('cycle report on engine-validated transcript uses real model analysis without live writes',async({page},info)=>{
 const fixture=process.env.CYCLE_FIXTURE??'/home/jihua/quoridor-backups/cycle-diagnosis-20261030/reconstructed.json'
 test.skip(!existsSync(fixture),'Set CYCLE_FIXTURE to the external engine-validated transcript')
 const data=JSON.parse(readFileSync(fixture,'utf8'))
 let steps=0,analyzes=0
 page.on('request',r=>{if(r.url().endsWith('/step'))steps++;if(r.url().endsWith('/api/analyze'))analyzes++})
 await page.route('**/api/agents/play/cycle-fixture',r=>r.fulfill({json:{position:data.positions.at(-1),positions:data.positions.slice(0,-1),identities:data.identities,actions:data.actions,revision:73,winner:null,limit:false,legal_actions:[]}}))
 await page.goto('/play?season=cnn-architecture-20261021-213018&p1=A-baseline_seed-44&p2=A-baseline_seed-43&game=cycle-fixture')
 await page.getByRole('button',{name:'循环诊断',exact:true}).click()
 await expect(page.locator('.aw-cycle')).toContainText('4 手一圈 · 9 圈')
 await expect(page.locator('.aw-cycle')).toContainText('所有相位的实际落子均与复算 Top1 一致')
 expect(analyzes).toBe(4);expect(steps).toBe(0)
 await expect(page.locator('.aw-cycle')).toContainText('policy_logits')
 await page.getByRole('button',{name:'查看该步',exact:true}).nth(2).click()
 await expect(page.getByLabel('决策依据')).toContainText('历史实际动作')
 expect(analyzes).toBe(4)
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'导出证据'}).click();expect((await download).suggestedFilename()).toBe('cycle-evidence.json')
 await page.screenshot({path:info.outputPath('cycle-diagnosis.png'),fullPage:true})
 await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true)
 await page.getByRole('button',{name:'返回对局',exact:true}).click()
 await expect(page.getByRole('button',{name:'Agent 下一手'})).toBeEnabled()
 expect(steps).toBe(0)
})
