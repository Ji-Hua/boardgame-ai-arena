import {test,expect} from '@playwright/test'
const season='cnn-architecture-20261021-213018'
test('dark workbench routes, responsive layout and saved theme',async({page},info)=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message))
 const routes=[['ladder','/ladder?season='+season],['agent','/agents/minimax_d2?season='+season],['training','/training?season='+season+'&run=40f0edbd60314c7ab822b6dfbd8522ef&run=9c1b3ed4a5004b5c933bb4bf77b594fe'],['replay','/games/049f457b806f4604720d%3A0?season='+season+'&agent=A-baseline_seed-44&opponent=greedy&ply=14'],['play','/play?season='+season+'&p1=minimax_d2&p2=human']]
 for(const [name,url] of routes){
  await page.setViewportSize({width:1440,height:1000});await page.goto(url)
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
  await expect(page.locator('.aw-main h1')).toBeVisible()
  if(name==='training')await expect(page.locator('.aw-training-chart svg path')).toHaveCount(2)
  if(name==='replay'){
   await expect(page.getByLabel('决策依据')).toContainText('当前代码复算')
   expect(await page.getByRole('img',{name:'Recorded Quoridor position'}).evaluate(e=>getComputedStyle(e).backgroundColor)).toBe('rgb(14, 24, 37)')
  }
  expect(await page.locator('.agent-workspace').evaluate(e=>getComputedStyle(e).backgroundColor)).toBe('rgb(11, 16, 24)')
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true)
  await page.screenshot({path:info.outputPath(name+'-dark.png'),fullPage:true})
  await page.setViewportSize({width:390,height:844})
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true)
  await page.screenshot({path:info.outputPath(name+'-mobile.png'),fullPage:true})
 }
 await page.getByRole('button',{name:'浅色',exact:true}).click()
 await page.reload();await expect(page.locator('html')).toHaveAttribute('data-theme','light')
 await page.screenshot({path:info.outputPath('play-light-mobile.png'),fullPage:true})
 await page.getByRole('button',{name:'深色',exact:true}).click()
 await page.reload();await expect(page.locator('html')).toHaveAttribute('data-theme','dark')
 expect(errors).toEqual([])
})
