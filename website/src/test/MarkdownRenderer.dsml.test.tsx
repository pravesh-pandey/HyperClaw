// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import MarkdownRenderer from '../components/MarkdownRenderer'

const DSML_FUNCTION_CALL = [
  '<｜DSML｜function_calls>',
  '<｜DSML｜invoke name="read_file">',
  '<｜DSML｜parameter name="path">README.md',
  '<｜DSML｜/parameter>',
  '<｜DSML｜/invoke>',
  '<｜DSML｜/function_calls>',
].join('')

describe('MarkdownRenderer — DSML protocol leakage', () => {
  it('hides a complete function-call envelope while preserving surrounding prose', () => {
    const { container } = render(
      <MarkdownRenderer content={`Before the call\n\n${DSML_FUNCTION_CALL}\n\nAfter the call`} />,
    )
    const text = container.textContent ?? ''
    expect(text).toContain('Before the call')
    expect(text).toContain('After the call')
    expect(text).not.toContain('DSML')
    expect(text).not.toContain('README.md')
  })

  it('suppresses an incomplete streamed function-call token', () => {
    const { container } = render(
      <MarkdownRenderer content={'Before\n<｜DSML｜function_calls'} streaming />,
    )
    expect(container.textContent ?? '').toBe('Before')
  })

  it('keeps authored DSML syntax visible inside inline code', () => {
    const { container } = render(
      <MarkdownRenderer content={'Document the token `<｜DSML｜function_calls>` here.'} />,
    )
    expect(container.textContent ?? '').toContain('<｜DSML｜function_calls>')
  })
})
