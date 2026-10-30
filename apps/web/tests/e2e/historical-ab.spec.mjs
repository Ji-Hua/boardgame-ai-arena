import {test,expect} from '@playwright/test'
import fs from 'node:fs'
const season='cnn-architecture-20261021-213018'
const fixture='/home/jihua/quoridor-backups/cycle-diagnosis-20261030/reconstructed.json'
test('historical A/B uses real model evidence without mutating a game',async({page},info)=>{
 test.skip(!fs.existsSync(fixture),'External historical fixture unavailable')
 const f=JSON.parse(fs.readFileSync(fixture,'utf8')),mutations=[],errors=[],branches=[]
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().includes('/api/agents/play')&&r.method()!=='GET')mutations.push(r.url());if(r.url().endsWith('/api/branch'))branches.push(r.postDataJSON())})
 await page.route('**/api/agents/play/history-fixture',route=>route.fulfill({json:{id:'history-fixture',identities:f.identities,position:f.positions[73],positions:f.positions.slice(0,73),actions:f.actions,revision:73,winner:null,limit:false,legal_actions:[]}}))
 await page.goto('/play?'+new URLSearchParams({season,game:'history-fixture'}))
 await page.getByText('走子记录',{exact:true}).click()
 await page.getByRole('button',{name:'40. P2 a2',exact:true}).click()
 await expect(page.getByLabel('诊断历史步')).toHaveValue('39')
 await expect(page.locator('.aw-actor-context')).toContainText(f.identities[1].name)
 const pair=page.getByLabel('动作 A B 比较');await expect(pair).toContainText('实际动作：a2')
 await page.getByLabel('对照动作 B').selectOption('pawn:2:1')
 await pair.getByRole('tab',{name:'输入证据',exact:true}).click()
 await expect(pair.locator('.lab-tensor')).toHaveCount(2)
 await expect(pair.locator('.lab-tensor svg')).toHaveCount(2)
 expect(branches.length).toBe(2);expect(branches[0].position).toEqual(f.positions[39]);expect(branches[0].action.target).toEqual([0,1]);expect(branches[1].action.target).toEqual([2,1]);expect(branches[0].model_id).toBe(f.identities[1].sha256)
 await expect(pair.locator('.aw-pair-scores')).toContainText('1.1945')
 await page.screenshot({path:info.outputPath('historical-input.png'),fullPage:true})
 await pair.getByRole('tab',{name:'分支比较',exact:true}).click();await pair.getByRole('button',{name:'查看 B 分支'}).click()
 await expect(page.locator('.aw-analysis>.aw-proof-note')).toContainText('假设分支 B')
 await pair.getByRole('tab',{name:'输入证据',exact:true}).click();await expect(page.locator('.aw-analysis>.aw-proof-note')).toContainText('原局面')
 expect(branches.length).toBe(2)
 await page.getByRole('tab',{name:'神经网络诊断',exact:true}).click();await expect(page.getByLabel('对照动作 B')).toHaveValue('pawn:2:1')
 await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true)
 await page.screenshot({path:info.outputPath('historical-mobile.png'),fullPage:true})
 await page.getByLabel('诊断历史步').selectOption('38');await expect(page.locator('.aw-actor-context')).toContainText(f.identities[0].name);await expect(pair).toContainText('实际动作：c6')
 await page.getByLabel('诊断历史步').selectOption('73');await expect(pair).toContainText('复算首选')
 expect(mutations).toEqual([]);expect(errors).toEqual([])
})
