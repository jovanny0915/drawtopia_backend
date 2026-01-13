// Supabase Edge Function for checking scheduled gifts (Cron Job)
// This function runs every minute to check for gifts that need to be delivered
// Deploy with: supabase functions deploy check-scheduled-gifts

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface Gift {
  id: string
  to_user_id: string
  delivery_time: string
  notification_sent: boolean
  status: string
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Initialize Supabase client with service role key
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? ''
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    // Get backend URL from environment
    const backendUrl = Deno.env.get('BACKEND_URL') || 'https://api.drawtopia.com'

    console.log('🔍 Checking for scheduled gifts to deliver...')

    const currentTime = new Date()
    const twoMinutesFromNow = new Date(currentTime.getTime() + 2 * 60 * 1000) // 2 minutes from now

    // Query gifts where:
    // 1. notification_sent is FALSE or NULL
    // 2. status is 'completed' (story generation is complete)
    // 3. delivery_time is within the next 2 minutes
    // 4. to_user_id is not null (user exists in system)
    const { data: gifts, error } = await supabase
      .from('gifts')
      .select('*')
      .eq('status', 'completed')
      .not('to_user_id', 'is', null)
      .or('notification_sent.is.null,notification_sent.eq.false')
      .lte('delivery_time', twoMinutesFromNow.toISOString())
      .gte('delivery_time', currentTime.toISOString())
      .order('delivery_time', { ascending: true })
      .limit(50) // Process max 50 gifts per run

    if (error) {
      console.error('❌ Error querying gifts:', error)
      throw error
    }

    if (!gifts || gifts.length === 0) {
      console.log('✅ No gifts to deliver at this time')
      return new Response(
        JSON.stringify({
          success: true,
          message: 'No gifts to deliver',
          processed: 0,
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        }
      )
    }

    console.log(`📦 Found ${gifts.length} gift(s) to deliver`)

    const results: any[] = []

    // Process each gift
    for (const gift of gifts) {
      try {
        console.log(`🎁 Processing gift ${gift.id} for user ${gift.to_user_id}`)

        // Call the backend API endpoint to trigger gift delivery
        const deliveryResponse = await fetch(`${backendUrl}/api/gift/deliver`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            gift_id: gift.id,
          }),
        })

        const deliveryResult = await deliveryResponse.json()

        if (deliveryResponse.ok && deliveryResult.success) {
          console.log(`✅ Gift ${gift.id} delivered successfully`)
          results.push({
            giftId: gift.id,
            success: true,
            message: 'Gift delivered successfully',
          })
        } else {
          console.error(`❌ Failed to deliver gift ${gift.id}:`, deliveryResult.error)
          results.push({
            giftId: gift.id,
            success: false,
            error: deliveryResult.error || 'Unknown error',
          })
        }
      } catch (error: any) {
        console.error(`❌ Error processing gift ${gift.id}:`, error)
        results.push({
          giftId: gift.id,
          success: false,
          error: error.message,
        })
      }
    }

    const successCount = results.filter(r => r.success).length
    const failureCount = results.filter(r => !r.success).length

    console.log(`✅ Processed ${results.length} gift(s): ${successCount} succeeded, ${failureCount} failed`)

    return new Response(
      JSON.stringify({
        success: true,
        processed: results.length,
        succeeded: successCount,
        failed: failureCount,
        results,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )
  } catch (error: any) {
    console.error('❌ Error in check-scheduled-gifts:', error)
    return new Response(
      JSON.stringify({
        success: false,
        error: error.message,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      }
    )
  }
})

