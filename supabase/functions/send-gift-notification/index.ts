// Supabase Edge Function for sending push notifications
// Deploy with: supabase functions deploy send-gift-notification

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import webpush from 'npm:web-push@3.6.7'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface PushSubscription {
  endpoint: string
  keys: {
    p256dh: string
    auth: string
  }
}

interface Gift {
  id: string
  to_user_id: string
  child_name: string
  occasion: string
  from_user_id: string
  delivery_time: string
}

// Web Push VAPID keys - Store these in Supabase Secrets
const VAPID_PUBLIC_KEY = Deno.env.get('VAPID_PUBLIC_KEY') || ''
const VAPID_PRIVATE_KEY = Deno.env.get('VAPID_PRIVATE_KEY') || ''
const VAPID_SUBJECT = Deno.env.get('VAPID_SUBJECT') || 'mailto:support@drawtopia.com'

// Initialize web-push with VAPID details
webpush.setVapidDetails(
  VAPID_SUBJECT,
  VAPID_PUBLIC_KEY,
  VAPID_PRIVATE_KEY
)

/**
 * Send a Web Push notification using web-push library
 */
async function sendWebPush(
  subscription: PushSubscription,
  payload: any
): Promise<{ success: boolean; error?: string; statusCode?: number }> {
  try {
    await webpush.sendNotification(
      subscription,
      JSON.stringify(payload),
      {
        TTL: 86400, // 24 hours
      }
    )
    return { success: true }
  } catch (error: any) {
    console.error('Error sending web push:', error)
    return { 
      success: false, 
      error: error.message,
      statusCode: error.statusCode 
    }
  }
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL') ?? ''
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    // Get request body
    const { giftId, mode } = await req.json()

    // Mode can be 'single' (for specific gift) or 'batch' (for all scheduled)
    const gifts: Gift[] = []

    if (mode === 'single' && giftId) {
      // Send notification for a specific gift
      const { data: gift, error } = await supabase
        .from('gifts')
        .select('*')
        .eq('id', giftId)
        .single()

      if (error) throw error
      if (gift) gifts.push(gift)
    } else {
      // Batch mode: Process all scheduled gifts
      const { data: scheduledGifts, error } = await supabase
        .from('gifts')
        .select('*')
        .eq('notification_scheduled', true)
        .eq('status', 'completed')
        .not('to_user_id', 'is', null)
        .lte('delivery_time', new Date().toISOString())
        .or('notification_sent.is.null,notification_sent.eq.false')
        .limit(50)

      if (error) throw error
      if (scheduledGifts) gifts.push(...scheduledGifts)
    }

    const results: any[] = []

    // Process each gift
    for (const gift of gifts) {
      try {
        // to_user_id might be stored as email or UUID - try to get user ID
        let recipientUserId = gift.to_user_id
        
        // If to_user_id looks like an email, try to find the user ID
        if (gift.to_user_id && gift.to_user_id.includes('@')) {
          const { data: userData } = await supabase.auth.admin.listUsers()
          const user = userData.users?.find(u => u.email?.toLowerCase() === gift.to_user_id.toLowerCase())
          recipientUserId = user?.id || gift.to_user_id
        }
        
        // Get push subscriptions for the recipient
        const { data: subscriptions, error: subError } = await supabase
          .from('push_subscriptions')
          .select('*')
          .eq('user_id', recipientUserId)

        if (subError) {
          console.error('Error fetching subscriptions:', subError)
          continue
        }
        
        if (!subscriptions || subscriptions.length === 0) {
          console.log(`No push subscriptions found for user ${recipientUserId}`)
          // Still mark as sent since we tried
          await supabase
            .from('gifts')
            .update({
              notification_sent: true,
              notification_sent_at: new Date().toISOString(),
            })
            .eq('id', gift.id)
          continue
        }

        // Get sender information
        const { data: senderData } = await supabase.auth.admin.getUserById(gift.from_user_id)
        const senderName = senderData?.user?.user_metadata?.name || 
                          senderData?.user?.email?.split('@')[0] || 
                          'Someone'

        // Prepare notification payload
        const notificationPayload = {
          title: '🎁 New Gift Story Received!',
          body: `${senderName} sent you a story for ${gift.child_name}`,
          icon: '/icon-192.png',
          badge: '/badge-72.png',
          data: {
            giftId: gift.id,
            url: `/gift/view/${gift.id}`,
            occasion: gift.occasion,
            childName: gift.child_name,
          },
          actions: [
            {
              action: 'view',
              title: 'View Gift',
            },
            {
              action: 'close',
              title: 'Close',
            },
          ],
        }

        // Send notification to all user's devices
        let sentCount = 0
        const failedEndpoints: string[] = []

        for (const subscription of subscriptions || []) {
          try {
            const pushSubscription: PushSubscription = {
              endpoint: subscription.endpoint,
              keys: {
                p256dh: subscription.p256dh,
                auth: subscription.auth,
              },
            }

            // Send push notification using web-push library
            const result = await sendWebPush(pushSubscription, notificationPayload)

            if (result.success) {
              sentCount++
            } else {
              failedEndpoints.push(subscription.endpoint)
              // If endpoint is invalid (410 Gone or 404), remove subscription
              if (result.statusCode === 410 || result.statusCode === 404) {
                await supabase
                  .from('push_subscriptions')
                  .delete()
                  .eq('endpoint', subscription.endpoint)
                console.log(`Removed invalid subscription: ${subscription.endpoint}`)
              }
            }
          } catch (error) {
            console.error('Error sending to endpoint:', error)
            failedEndpoints.push(subscription.endpoint)
          }
        }

        // Mark notification as sent
        await supabase
          .from('gifts')
          .update({
            notification_sent: true,
            notification_sent_at: new Date().toISOString(),
          })
          .eq('id', gift.id)

        results.push({
          giftId: gift.id,
          success: true,
          sentCount,
          failedEndpoints,
        })
      } catch (error) {
        console.error(`Error processing gift ${gift.id}:`, error)
        results.push({
          giftId: gift.id,
          success: false,
          error: error.message,
        })
      }
    }

    return new Response(
      JSON.stringify({
        success: true,
        processed: gifts.length,
        results,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )
  } catch (error) {
    console.error('Error in send-gift-notification:', error)
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


